from fastapi import UploadFile
import fitz  # PyMuPDF
import uuid as _uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from qdrant_client.models import PointStruct

from src.config import COLLECTION_NAME
from src.embeddings import get_embeddings
from src.vectorstores import get_qdrant_client, init_qdrant


async def ingest_pdf(file: UploadFile, collection_name: str = None):
    target_collection = collection_name or COLLECTION_NAME

    print(f"📄 Processing PDF file: {file.filename}")
    content = await file.read()

    docs = []
    pdf  = fitz.open(stream=content, filetype="pdf")
    page_count = len(pdf)
    print(f"📑 PDF loaded successfully. Found {page_count} pages")

    try:
        for page_num in range(page_count):
            page = pdf[page_num]
            text = page.get_text()
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"page": page_num, "source": file.filename}
                ))
    finally:
        pdf.close()

    print("✂️ Splitting document into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"📚 Created {len(chunks)} text chunks")

    print("🧮 Generating embeddings...")
    try:
        texts      = [chunk.page_content for chunk in chunks]
        embeddings = get_embeddings(texts)
        print(f"✅ Generated {len(embeddings)} embeddings")
    except Exception as e:
        print(f"❌ Embedding error: {type(e).__name__}: {e}")
        raise

    init_qdrant(target_collection)
    client = get_qdrant_client()

    points = []
    for chunk, embedding in zip(chunks, embeddings):
        point_id = int(_uuid.uuid4().int >> 64)
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text":   chunk.page_content,
                    "page":   chunk.metadata.get("page", 0),
                    "source": chunk.metadata.get("source", file.filename),
                }
            )
        )

    print(f"⬆️ Uploading {len(points)} documents to '{target_collection}'...")
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=target_collection,
            points=points[i: i + batch_size],
            wait=True
        )
        print(f"   Uploaded batch {i // batch_size + 1}")

    print(f"✅ Upload complete! Added {len(chunks)} chunks from {page_count} pages")