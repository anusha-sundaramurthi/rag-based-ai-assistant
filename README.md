<div align="center">

# 🧭 AI Travel Assistant

**A RAG-based travel planning assistant. Upload travel guide PDFs, then ask for itineraries, hotels, food and activities in your own language.**

[**Live Demo →**](https://travel-assistant-yphb.onrender.com/)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C)
![Qdrant](https://img.shields.io/badge/Qdrant-DC244C)

![Deployed on Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?logo=render&logoColor=white)

</div>

---

## Overview

Generic chatbots answer travel questions from whatever they absorbed during training. **AI Travel Assistant** answers from the guides *you* provide. You upload travel guide PDFs, the app indexes them in a vector database, and every answer is grounded in the retrieved passages from your documents. It can also fall back to general knowledge when you have not uploaded anything relevant.

The assistant keeps conversational context, so follow-ups like *"what about food there?"* work without repeating the destination. Replies can be generated in 12 languages.

## Features

- 📄 **Upload travel guide PDFs.** Drag and drop one or more guides and process them with one click.
- 🔎 **Retrieval-Augmented Generation.** Answers are built from the most relevant chunks of your uploaded guides.
- 💾 **Persistent PDF memory.** Indexed guides are stored in Qdrant, so you do not need to re-upload them each session.
- 💬 **Context-aware chat.** Follow-up questions understand what you were just discussing.
- 🌐 **Multilingual responses.** English, Tamil, Hindi, Telugu, Malayalam, Kannada, French, German, Spanish, Japanese, Chinese and Arabic.
- 🧠 **General-knowledge mode.** Optionally answer beyond the uploaded documents (`use_general`).
- 🧩 **Embeddable widget support.** Each widget gets its own isolated document collection and API key, managed through an admin page.
- 🗑️ **Simple controls.** Clear the chat without touching your PDFs, or wipe the whole database.

## Live Demo

**https://travel-assistant-yphb.onrender.com/**

> The app is hosted on Render. If it has been idle, the first request may take a short while to wake the server.

Try questions like:

- *Plan a 5-day trip to Bali*
- *Best beaches in Thailand*
- *Hotels in Paris under $150*
- *Food guide for Tokyo*

## How It Works

```
 PDF guides ──► Text extraction ──► Chunking ──► Embeddings ──► Qdrant
  (PyMuPDF)     (PyMuPDF)        (LangChain     (NVIDIA NIM)   vector store
                                  splitters)

 User question ──► Embed query ──► Retrieve top chunks ──► LLM (NVIDIA NIM) ──► Answer
        ▲                                │                       ▲
        │                         Chat history (per session)  Selected language
        └────────────────── follow-up questions ─────────────────┘
```

1. **Ingest.** Uploaded PDFs are parsed with PyMuPDF, split into chunks, embedded and stored in Qdrant.
2. **Retrieve.** Each question is embedded and matched against the stored chunks.
3. **Generate.** The retrieved context, session chat history and chosen language are passed to the LLM, which writes the answer.

## Tech Stack

| Layer | Technology |
| --- | --- |
| Backend | Python 3.11, FastAPI, Uvicorn |
| RAG orchestration | LangChain (`langchain`, `langchain-core`, `langchain-community`, `langchain-text-splitters`) |
| LLM / embeddings | NVIDIA NIM via the OpenAI-compatible client (`langchain-openai`, `openai`) |
| Vector database | Qdrant (`qdrant-client`) |
| PDF parsing | PyMuPDF |
| Frontend | Static HTML/CSS/JS served by FastAPI |
| Deployment | Render |

## Project Structure

```
rag-based-ai-assistant/
├── src/              # FastAPI application and RAG logic
├── static/           # Frontend assets
├── requirements.txt  # Python dependencies
├── runtime.txt       # Python version for deployment (3.11.0)
└── .gitignore
```

## Getting Started

### Prerequisites

- Python **3.11**
- An **Groq** API key
- A **Qdrant** instance (Qdrant Cloud or a local Docker container)

### 1. Clone the repository

```bash
git clone https://github.com/anusha-sundaramurthi/rag-based-ai-assistant.git
cd rag-based-ai-assistant
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
# Groq
GROQ_API_KEY=your_groq_api_key

# Qdrant
QDRANT_URL=https://your-qdrant-host:6333
QDRANT_API_KEY=your_qdrant_api_key

# Admin access (protects the /admin endpoints)
ADMIN_API_KEY=choose_a_strong_secret
```

> Never commit your `.env` file. It is already covered by `.gitignore`.

To run Qdrant locally instead of using the cloud:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Run the app

```bash
uvicorn src.main:app --reload
```

Open **http://localhost:8000** in your browser.

## Usage

1. **Upload guides.** Click *Upload Guides* in the sidebar (or drag PDFs onto it), then click **⚡ Process Guides** and wait for the confirmation toast.
2. **Pick a language** from the sidebar.
3. **Ask a question** in the chat box. Press `Enter` to send and `Shift + Enter` for a new line.
4. **Ask follow-ups.** The assistant remembers the conversation within your session.

| Button | What it does | PDFs kept? |
| --- | --- | --- |
| 🗑️ Clear Chat | Clears chat messages only | ✅ Yes |
| 🗄️ Clear DB | Deletes all uploaded PDFs from the vector store | ❌ No |

## API Reference

FastAPI generates interactive documentation at `/docs` (Swagger UI).

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Serves the web app |
| `POST` | `/upload-default` | Upload a PDF to the default collection (used by the main UI) |
| `POST` | `/ask-default` | Ask a question against the default collection |
| `POST` | `/upload` | Upload a PDF to a specific widget's collection (`x-api-key` header) |
| `POST` | `/ask` | Chat endpoint for embedded widgets, scoped by `widget_id` |
| `POST` | `/clear` | Clear chat history for a `session_id` (and optional `widget_id`) |
| `POST` | `/clear-db` | Delete all stored documents |
| `GET` | `/admin` | Admin page |
| `GET` / `POST` | `/admin/widgets` | List or create widgets (`x-api-key` header) |
| `DELETE` | `/admin/widgets/{widget_id}` | Delete a widget (`x-api-key` header) |

### Example request

```bash
curl -X POST http://localhost:8000/ask-default \
  -H "Content-Type: application/json" \
  -d '{
        "query": "Plan a 5-day trip to Bali",
        "session_id": "demo-session",
        "language": "English",
        "use_general": false
      }'
```

**Request body fields**

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `query` | string | required | The user's question |
| `session_id` | string | `"default"` | Keeps conversation history separate per user |
| `language` | string | `"English"` | Language of the reply |
| `use_general` | boolean | `false` | Allow answers beyond the uploaded documents |

## Deployment

The project is deployed on [Render](https://render.com/) as a Python web service.

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn src.main:app --host 0.0.0.0 --port $PORT`
- **Environment variables:** add the same variables from your `.env` file in the Render dashboard.
- `runtime.txt` pins the Python version to 3.11.0.

## Roadmap

- [ ] Source citations showing which PDF and page each answer came from
- [ ] Support for more document types (DOCX, web pages)
- [ ] Streaming responses
- [ ] Saved and exportable itineraries

## Contributing

Contributions, issues and feature requests are welcome. Feel free to open an issue or submit a pull request.

## Author

**Anusha Sundaramurthi**
GitHub: [@anusha-sundaramurthi](https://github.com/anusha-sundaramurthi)
