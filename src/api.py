import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import setup_logging
from src.data.ingestion import download_dataset
from src.data.preprocessing import get_unique_cuisines, get_unique_locations, preprocess
from src.input.validator import validate_and_normalize
from src.recommendation.engine import get_recommendations

# Setup logging
setup_logging()
logger = logging.getLogger("zomato_recommender.api")

app = FastAPI(title="The Last Bite — AI Dining Concierge API")

# Add CORS middleware to allow the frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dataset variable
df = None

# Cached location/cuisine lists (computed once after data load)
_cached_locations = None
_cached_cuisines = None

def load_data():
    global df, _cached_locations, _cached_cuisines
    if df is not None:
        return
    try:
        logger.info("Loading Zomato dataset...")
        start = time.perf_counter()
        
        candidates = [
            Path(__file__).resolve().parent.parent / "data" / "restaurants_clean.parquet",
            Path.cwd() / "data" / "restaurants_clean.parquet",
            Path("/var/task/data/restaurants_clean.parquet"),
        ]
        clean_path = None
        for p in candidates:
            if p.exists():
                clean_path = p
                break

        if clean_path:
            logger.info(f"Loading preprocessed dataset from {clean_path}...")
            df = pd.read_parquet(clean_path)
        else:
            raw_df = download_dataset()
            logger.info(f"Pre-processing {len(raw_df)} records...")
            df = preprocess(raw_df)

        # Normalize cuisines_list if read from parquet (convert ndarray to list of strings)
        if "cuisines_list" in df.columns:
            df["cuisines_list"] = df["cuisines_list"].apply(
                lambda x: [str(c).strip() for c in x if c and str(c).strip() and str(c).lower() != "nan"]
                if hasattr(x, "__iter__") and not isinstance(x, str)
                else []
            )

        # Pre-compute and cache location/cuisine lists with Bangalore as default option
        raw_locs = get_unique_locations(df)
        _cached_locations = ["Bangalore"] + [l for l in raw_locs if l.lower() != "bangalore"]
        _cached_cuisines = get_unique_cuisines(df)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Data loaded successfully: {len(df)} records, {len(_cached_locations)} locations, {len(_cached_cuisines)} cuisines ready in {elapsed:.0f}ms.")
    except Exception:
        logger.exception("Error loading dataset")
        raise

@app.on_event("startup")
def on_startup():
    load_data()

def ensure_data_loaded():
    if df is None:
        load_data()

# Mount static directory for the React SPA (supports root, public, and frontend)
_project_root = Path(__file__).resolve().parent.parent
_static_dir = _project_root / "public" / "static" if (_project_root / "public" / "static").exists() else _project_root / "frontend"
_index_file = _project_root / "public" / "index.html" if (_project_root / "public" / "index.html").exists() else _project_root / "frontend" / "index.html"

app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

@app.get("/")
def read_root():
    return FileResponse(str(_index_file))

class RecommendRequest(BaseModel):
    location: str = "Bangalore"
    budget: int | None = None
    party_size: int = 2
    cuisine: str | None = None
    min_rating: float = 3.5
    preferences: list[str] = Field(default_factory=list)

@app.get("/api/health")
def health():
    ensure_data_loaded()
    return {"status": "ok", "records_loaded": len(df) if df is not None else 0}

@app.get("/api/locations")
def get_locations():
    ensure_data_loaded()
    if _cached_locations is None:
        raise HTTPException(status_code=500, detail="Data not loaded yet")
    return _cached_locations

@app.get("/api/cuisines")
def get_cuisines():
    ensure_data_loaded()
    if _cached_cuisines is None:
        raise HTTPException(status_code=500, detail="Data not loaded yet")
    return _cached_cuisines

@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    ensure_data_loaded()
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded yet")
    
    req_start = time.perf_counter()
    
    try:
        validated, warnings = validate_and_normalize(
            location=req.location,
            budget=req.budget,
            party_size=req.party_size,
            cuisine=req.cuisine,
            min_rating=req.min_rating,
            preferences=req.preferences,
            known_locations=_cached_locations,
            known_cuisines=_cached_cuisines
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    try:
        response = get_recommendations(df, validated)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    result = response.get("result")
    if result is None or not result.recommendations:
        return {
            "success": False,
            "message": f"No restaurants found matching your criteria in {validated.location}.",
            "warnings": warnings,
            "recommendations": [],
            "summary": "",
            "candidates_found": 0,
            "response_time_ms": response.get("response_time_ms", 0)
        }
        
    # Convert recommendations to a list of dicts and enrich them
    import re
    recs = []
    for r in result.recommendations:
        votes = 0
        cost_val = 0
        online_order = False
        book_table = True  # Enable Concierge table reservation for recommended venues
        restaurant_type = "Casual Dining"
        
        # Look up in dataset to extract actual votes, cost, and options
        clean_name = r.restaurant_name.strip().lower()
        matches = df[df['restaurant_name'].astype(str).str.strip().str.lower() == clean_name]
        
        # If no exact match, try substring match
        if matches.empty:
            matches = df[df['restaurant_name'].astype(str).str.lower().str.contains(re.escape(clean_name), na=False)]
            
        if not matches.empty:
            loc_matches = matches[matches['location'].astype(str).str.strip().str.lower() == validated.location.strip().lower()]
            row = loc_matches.iloc[0] if not loc_matches.empty else matches.iloc[0]
            raw_votes = row.get('votes', 0)
            votes = int(raw_votes) if pd.notna(raw_votes) else 0

            raw_cost = row.get('cost_for_two', 0)
            cost_val = int(raw_cost) if pd.notna(raw_cost) else 0

            online_order = bool(row.get('online_order', False)) if pd.notna(row.get('online_order')) else False
            restaurant_type = str(row.get('restaurant_type', 'Casual Dining'))
            
        # If cost_val is still 0, parse digits from r.estimated_cost
        if cost_val <= 0 and r.estimated_cost:
            digits = re.findall(r'\d+', r.estimated_cost)
            if digits:
                cost_val = int(digits[0])
        if cost_val <= 0:
            cost_val = validated.budget or 800

        if votes <= 0:
            votes = 120

        recs.append({
            "rank": r.rank,
            "restaurant_name": r.restaurant_name,
            "cuisine": r.cuisine,
            "rating": r.rating,
            "estimated_cost": r.estimated_cost if r.estimated_cost and "Rs" in r.estimated_cost or "₹" in r.estimated_cost else f"₹{cost_val} for two",
            "explanation": r.explanation,
            "votes": votes,
            "cost_value": cost_val,
            "online_order": online_order,
            "book_table": book_table,
            "restaurant_type": restaurant_type
        })
        
    total_ms = int((time.perf_counter() - req_start) * 1000)
    logger.info(f"Recommendation served: {len(recs)} results in {total_ms}ms for {validated.location}")
    
    return {
        "success": True,
        "warnings": warnings,
        "recommendations": recs,
        "summary": result.summary,
        "candidates_found": response.get("candidates_found", 0),
        "response_time_ms": response.get("response_time_ms", total_ms),
        "model_used": response.get("model_used", "unknown")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=True)
