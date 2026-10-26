from contextlib import asynccontextmanager
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from refund_agent.api.config import settings
from refund_agent.api.routes.chat import router as chat_router
from refund_agent.api.routes.customers import router as customers_router
from refund_agent.api.routes.health import router as health_router
from refund_agent.api.routes.policy import router as policy_router
from refund_agent.api.routes.threads import router as threads_router

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
EVAL_REPORT_PATH = Path(__file__).resolve().parents[1] / "eval_report.pdf"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup tasks
    print(f"Starting {settings.app_name} v{settings.app_version}...")
    yield
    # Shutdown tasks
    print(f"Shutting down {settings.app_name}...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": str(exc),
            "path": request.url.path,
        },
    )


# Include API Routers
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(threads_router)
app.include_router(customers_router)
app.include_router(policy_router)

# Mount Static Files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", tags=["Frontend"])
def serve_frontend():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Frontend index.html not found"}


@app.get("/api/eval-report", tags=["Evaluation"])
def get_eval_report():
    """Serve the evaluation report PDF inline for browser viewing."""
    if not EVAL_REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="eval_report.pdf not found.")
    return FileResponse(
        EVAL_REPORT_PATH,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline; filename=eval_report.pdf"},
    )


@app.get("/api/info", tags=["Root"])
def api_info():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": settings.app_description,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "endpoints": {
            "chat": "POST /api/chat",
            "chat_stream": "POST /api/chat/stream",
            "threads": "GET /api/threads",
            "thread_state": "GET /api/threads/{thread_id}/state",
            "thread_history": "GET /api/threads/{thread_id}/history",
            "delete_thread": "DELETE /api/threads/{thread_id}",
            "customers": "GET /api/customers",
            "customer_detail": "GET /api/customers/{username}",
            "policy": "GET /api/policy",
            "health": "GET /health",
            "eval_report": "GET /api/eval-report",
        },
    }


def start():
    """CLI entrypoint to run the server."""
    uvicorn.run(
        "refund_agent.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    start()
