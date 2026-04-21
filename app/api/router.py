"""Aggregate API routers for the FastAPI application."""

from fastapi import APIRouter

from app.api.routes import assistant, generation, health, history, projects

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(assistant.router)
api_router.include_router(projects.router)
api_router.include_router(generation.router)
api_router.include_router(history.router)
