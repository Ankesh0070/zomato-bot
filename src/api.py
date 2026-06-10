import sys
import time
import logging
from pathlib import Path
from functools import lru_cache
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd

from src.data.ingestion import download_dataset
from src.data.preprocessing import preprocess, get_unique_locations, get_unique_cuisines
from src.input.validator import validate_and_normalize
from src.recommendation.engine import get_recommendations
from src.config import setup_logging

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

@app.on_event("startup")
def load_data():
    global df, _cached_locations, _cached_cuisines
    try:
        logger.info("Loading Zomato dataset...")
        start = time.perf_counter()
        raw_df = download_dataset()
        logger.info(f"Pre-processing {len(raw_df)} records...")
        df = preprocess(raw_df)
        # Pre-compute and cache location/cuisine lists
        _cached_locations = get_unique_locations(df)
        _cached_cuisines = get_unique_cuisines(df)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Data loaded successfully: {len(df)} records ready in {elapsed:.0f}ms.")
    except Exception as e:
        logger.exception(f"Error loading dataset: {e}")
        raise e

# Mount the static directory for the React SPA
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def read_root():
    return FileResponse("frontend/index.html")

class RecommendRequest(BaseModel):
    location: str
    budget: Optional[int] = None
    party_size: int = 2
    cuisine: Optional[str] = None
    min_rating: float = 3.5
    preferences: list[str] = []

@app.get("/api/health")
def health():
    return {"status": "ok", "records_loaded": len(df) if df is not None else 0}

@app.get("/api/locations")
def get_locations():
    if _cached_locations is None:
        raise HTTPException(status_code=500, detail="Data not loaded yet")
    return _cached_locations

@app.get("/api/cuisines")
def get_cuisines():
    if _cached_cuisines is None:
        raise HTTPException(status_code=500, detail="Data not loaded yet")
    return _cached_cuisines

@app.post("/api/recommend")
def recommend(req: RecommendRequest):
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
    recs = []
    for r in result.recommendations:
        votes = 0
        cost_val = 0
        online_order = False
        book_table = False
        restaurant_type = "Casual Dining"
        
        # Look up in dataset to extract actual votes, cost, and options
        matches = df[df['restaurant_name'].str.lower() == r.restaurant_name.lower()]
        if not matches.empty:
            loc_matches = matches[matches['location'].str.lower() == validated.location.lower()]
            row = loc_matches.iloc[0] if not loc_matches.empty else matches.iloc[0]
            votes = int(row.get('votes', 0))
            cost_val = int(row.get('cost_for_two', 0))
            online_order = bool(row.get('online_order', False))
            book_table = bool(row.get('book_table', False))
            restaurant_type = str(row.get('restaurant_type', 'Casual Dining'))
            
        recs.append({
            "rank": r.rank,
            "restaurant_name": r.restaurant_name,
            "cuisine": r.cuisine,
            "rating": r.rating,
            "estimated_cost": r.estimated_cost,
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
