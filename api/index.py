import sys
from pathlib import Path

# Add project root directory to sys.path so 'src' can be imported anywhere
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import traceback

try:
    from src.api import app
except Exception as e:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Error Fallback")
    _err_tb = traceback.format_exc()

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    def catch_all(path: str):
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": _err_tb,
                "sys_path": sys.path,
                "cwd": str(Path.cwd()),
                "files_in_cwd": [str(p) for p in Path.cwd().rglob("*")][:100],
            },
        )

