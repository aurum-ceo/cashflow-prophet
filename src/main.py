"""
FastAPI entry point for CashFlow Prophet.
"""
import uvicorn
from fastapi import FastAPI
from src.api import routes

app = FastAPI(
    title="CashFlow Prophet API",
    description="Pronóstico y optimización de flujo de caja.",
    version="0.1.0",
)

# Register API router
app.include_router(routes.router, prefix="/api/v1")

@app.get("/", tags=["root"])
async def root() -> dict:
    return {
        "message": "Bienvenido a CashFlow Prophet API",
        "docs": "/docs",
        "redoc": "/redoc",
        "status": "ready",
    }

@app.get("/health", tags=["health"])
async def health() -> dict:
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
