import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.widget_store import (
    verify_master_key, create_widget, get_widget,
    get_all_widgets, add_pdf, delete_widget,
    verify_widget_key
)
from src.generator import generate_answer, clear_memory
from src.ingest import ingest_pdf
from src.vectorstores import init_qdrant, clear_qdrant, init_bot_collection, delete_collection


# ── Startup ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_init_db())
    yield

async def _init_db():
    try:
        print("Initializing Qdrant database...")
        init_qdrant()
        print("Database initialization complete.")
    except Exception as e:
        print(f"Qdrant init error: {e}")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ──────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/admin")
def admin_page():
    return FileResponse("static/admin.html")


# ── Auth helper ───────────────────────────────────────────
def require_master(x_api_key: str = Header(None)):
    if not x_api_key or not verify_master_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid master API key")
    return x_api_key


# ════════════════════════════════════════════════════════
# ADMIN ROUTES — protected by master API key
# ════════════════════════════════════════════════════════

@app.get("/admin/widgets")
async def admin_list_widgets(x_api_key: str = Header(None)):
    require_master(x_api_key)
    return get_all_widgets()


@app.post("/admin/widgets")
async def admin_create_widget(
    request: Request,
    x_api_key: str = Header(None)
):
    require_master(x_api_key)
    body = await request.json()

    name   = body.get("name", "").strip()
    config = body.get("config", {})

    if not name:
        raise HTTPException(status_code=400, detail="Widget name is required")
    if not config.get("business_context", "").strip():
        raise HTTPException(status_code=400, detail="Business context is required")

    widget = create_widget(name, config)
    init_bot_collection(widget["widget_id"])
    return widget


@app.delete("/admin/widgets/{widget_id}")
async def admin_delete_widget(
    widget_id: str,
    x_api_key: str = Header(None)
):
    require_master(x_api_key)
    widget = get_widget(widget_id)
    if not widget:
        raise HTTPException(status_code=404, detail="Widget not found")
    delete_collection(widget["collection"])
    delete_widget(widget_id)
    return {"success": True}


# ── Upload PDF to a widget (uses widget api_key) ──────────
@app.post("/upload")
async def upload_pdf(
    file: UploadFile = None,
    x_api_key: str = Header(None)
):
    """
    Upload PDF to a widget's collection.
    Uses the widget's own api_key (not master key).
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")

    # Find widget by api_key
    from src.widget_store import get_widget_by_api_key
    widget = get_widget_by_api_key(x_api_key)
    if not widget:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not file:
        return {"message": "No file uploaded"}
    if not file.filename.endswith(".pdf"):
        return {"message": "Only PDF files accepted"}

    try:
        await ingest_pdf(file, collection_name=widget["collection"])
        add_pdf(widget["widget_id"], file.filename)
        return {"message": f"{file.filename} processed successfully"}
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════
# WIDGET ASK ROUTE — used by embedded widget.js
# ════════════════════════════════════════════════════════

class AskRequest(BaseModel):
    query:       str
    session_id:  str  = "default"
    widget_id:   str  = ""
    language:    str  = "English"
    use_general: bool = False


@app.post("/ask")
async def ask(req: AskRequest):
    """
    Main chat endpoint used by the embedded widget.
    widget_id identifies which bot's collection to search.
    Falls back to default collection if no widget_id.
    """
    collection_name = None

    if req.widget_id:
        widget = get_widget(req.widget_id)
        if widget:
            collection_name = widget["collection"]
            # Prefix session with widget_id to avoid cross-widget memory
            session_id = f"{req.widget_id}_{req.session_id}"

            # Inject business context into language if set
            business_context = widget.get("config", {}).get("business_context", "")
        else:
            session_id = req.session_id
            business_context = ""
    else:
        session_id       = req.session_id
        business_context = ""

    result = generate_answer(
        req.query,
        session_id=session_id,
        use_general=req.use_general,
        language=req.language,
        collection_name=collection_name,
        business_context=business_context
    )

    return {
        "response":        result["answer"],
        "rewritten_query": result["rewritten_query"],
        "has_pdf_context": result["has_pdf_context"]
    }


@app.post("/clear")
async def clear_chat(session_id: str = "default", widget_id: str = ""):
    sid = f"{widget_id}_{session_id}" if widget_id else session_id
    clear_memory(sid)
    return {"message": "Session cleared"}


# ════════════════════════════════════════════════════════
# ORIGINAL ROUTES — your existing chat app
# ════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    query:       str
    session_id:  str  = "default"
    use_general: bool = False
    language:    str  = "English"

@app.post("/ask-default")
async def ask_default(req: QueryRequest):
    """Original chat route for your own index.html."""
    result = generate_answer(
        req.query,
        session_id=req.session_id,
        use_general=req.use_general,
        language=req.language
    )
    return {
        "response":        result["answer"],
        "rewritten_query": result["rewritten_query"],
        "has_pdf_context": result["has_pdf_context"]
    }

@app.post("/upload-default")
async def upload_default(file: UploadFile = None):
    """Original upload route for your own index.html."""
    if not file:
        return {"message": "No file uploaded"}
    if not file.filename.endswith(".pdf"):
        return {"message": "Please upload a PDF file"}
    try:
        await ingest_pdf(file)
        return {"message": "File processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-db")
async def clear_database():
    clear_qdrant()
    return {"message": "Database cleared."}
