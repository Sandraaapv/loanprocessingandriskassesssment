"""
Compatibility entrypoint for LoanGuard Underwriting API.
Redirects to the calibrated production FastAPI app in app.py.
"""
from app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
