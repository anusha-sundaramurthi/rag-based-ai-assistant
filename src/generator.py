from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

from src.retriever import retrieve_docs
from src.config import MODEL_PROVIDER, OPENAI_API_KEY
import os

# ── 1. Build the LLM ─────────────────────────────────────
if MODEL_PROVIDER == "openai":
    llm = ChatOpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        model_name="llama-3.3-70b-versatile",
        temperature=0.7)
else:
    raise ValueError(f"Unknown MODEL_PROVIDER: {MODEL_PROVIDER}")

# ── 2. Per-session memory store ───────────────────────────
_memory_store: dict[str, ConversationBufferMemory] = {}

def get_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in _memory_store:
        _memory_store[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    return _memory_store[session_id]

def clear_memory(session_id: str) -> None:
    if session_id in _memory_store:
        del _memory_store[session_id]

# ── 3. Language instruction ───────────────────────────────
def get_language_instruction(language: str) -> str:
    return (
        f"\n\nLANGUAGE INSTRUCTION:\n"
        f"You MUST respond ENTIRELY in {language}.\n"
        f"Every word of your answer must be in {language}.\n"
        f"Do not mix any other language.\n"
        f"Even if the question is in a different language, answer in {language}."
    )

def detect_language(text: str) -> str:
    """
    Auto-detects the language the user typed in.
    Returns a language name like 'Hindi', 'French', 'Tamil', etc.
    Falls back to 'English' on any error.
    """
    try:
        response = llm.invoke([
            {
                "role": "system",
                "content": (
                    "You are a language detector. "
                    "Reply with ONLY the English name of the language (e.g. Hindi, French, Tamil, Arabic, English). "
                    "No explanation. No punctuation. Just the single language name."
                )
            },
            {
                "role": "user",
                "content": f"What language is this text written in?\n\n{text}"
            }
        ])
        detected = response.content.strip().split()[0]  # take first word only
        print(f"[Generator] Detected language: '{detected}'")
        return detected if detected else "English"
    except Exception as e:
        print(f"[Generator] Language detection failed: {e}")
        return "English"

# ── 4. Domain label helper ────────────────────────────────
def _domain_label(business_context: str) -> str:
    if business_context and business_context.strip():
        return business_context.strip()
    return "a helpful AI assistant"

# ── 5. Dynamic prompt builders ────────────────────────────
# NOTE: All placeholders that LangChain will fill later use single braces:
#   {context}, {chat_history}, {language_instruction}, {question}
# The only f-string interpolation at build time is {domain} (Python variable).

def build_rewrite_prompt(business_context: str) -> str:
    domain = _domain_label(business_context)
    return (
        f"You are a query rewriter for {domain}.\n"
        "Your ONLY job is to rewrite the user's latest question so it is completely\n"
        "self-contained and explicit — replacing all vague pronouns and references\n"
        '(like "it", "that", "there", "both", "the first one", "those", "same one",\n'
        '"the previous", "that option", "this one")\n'
        "with the actual named entities found in the conversation history.\n"
        "Rules:\n"
        "- If the question references multiple items implicitly, name ALL of them explicitly.\n"
        "- If the question is already explicit and clear, return it UNCHANGED.\n"
        "- Return ONLY the rewritten question. No explanation. No extra text.\n"
        "- Do not answer the question. Just rewrite it.\n"
        "- Always rewrite in English regardless of input language.\n"
        "Conversation history:\n"
        "{chat_history}\n"
    )

def build_pdf_prompt(business_context: str) -> str:
    domain = _domain_label(business_context)
    return (
        f"You are an expert and friendly AI assistant for: {domain}.\n"
        "Use the following document context to answer the user's question.\n"
        "\n"
        "CRITICAL RULES:\n"
        "1. Read the user's question carefully and identify exactly what they are asking\n"
        "2. Read the context carefully\n"
        "3. Ask yourself: Does this context DIRECTLY match what the user is asking about?\n"
        "4. If YES → answer ONLY using the provided context\n"
        "5. You may improve grammar and sentence flow\n"
        "6. NEVER add facts, figures, names, or details not explicitly present in the context\n"
        "\n"
        "STRICT DOCUMENT MODE:\n"
        "- Use ONLY information explicitly present in the context\n"
        "- NEVER invent details, options, names, prices, or examples\n"
        "- If the document contains only limited information, give only limited information\n"
        "- Do NOT expand or fill gaps using your own general knowledge\n"
        "- Your job is to summarize document content, NOT generate new content\n"
        "\n"
        "VERY IMPORTANT FILTERING RULE:\n"
        "- Ignore unrelated paragraphs even if they appear in the same retrieved chunk\n"
        "- Extract ONLY sentences directly related to the user's question\n"
        "- Never summarize the whole chunk unless the user explicitly asks for all information\n"
        "\n"
        "IMPORTANT BEHAVIOR:\n"
        "- Use ONLY the provided document context\n"
        "- NEVER use outside/general knowledge\n"
        "- If partial relevant information exists, answer using ONLY that partial information\n"
        "- You may summarize and reorganize the content naturally\n"
        "- Return NO_PDF_CONTEXT ONLY if absolutely no relevant information exists\n"
        "\n"
        "IMPORTANT RULES FOR NO_PDF_CONTEXT:\n"
        "- Return ONLY the single word: NO_PDF_CONTEXT\n"
        "- No emoji, no explanation, no extra text — just: NO_PDF_CONTEXT\n"
        "- Do this ONLY when context has ZERO relevant info about what user asked\n"
        "\n"
        "FINAL STRICT RULE:\n"
        "- If a sentence does not directly answer the user's question, DO NOT include it\n"
        "- Prefer incomplete but accurate answers over extra unrelated information\n"
        "\n"
        "═══════════════════════════════════════════════════════════════\n"
        "ANSWER FORMAT INSTRUCTIONS:\n"
        "═══════════════════════════════════════════════════════════════\n"
        "\n"
        "STEP 1: IDENTIFY QUERY TYPE\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Detect what the user is asking and respond in the most natural format:\n"
        "- Overview / summary      → Short structured summary\n"
        "- List of items / options → Bullet list with brief descriptions\n"
        "- Step-by-step process    → Numbered steps\n"
        "- Comparison              → Side-by-side or categorized list\n"
        "- Single specific fact    → Direct one or two sentence answer\n"
        "- Detailed explanation    → Flowing paragraphs with headers if needed\n"
        "\n"
        "STEP 2: STRICT ANSWER RULES\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "GOLDEN RULE: Answer EXACTLY what user asked. Nothing more. Nothing less.\n"
        "- Ignore all unrelated information in the context\n"
        "- Extract and answer ONLY the parts directly relevant to the user's query\n"
        "- Never summarize the full chunk unless the user explicitly asked for all details\n"
        "\n"
        "WARNINGS — STRICT RULES:\n"
        "- ONLY add ⚠️ warning if user SPECIFICALLY asked for that info AND it is missing from context\n"
        "- If user did NOT ask for something → DO NOT mention it at all\n"
        "\n"
        "STEP 3: FORMAT BEAUTIFULLY (USER-FRIENDLY!)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "✅ Use bullet points (-) for lists\n"
        "✅ Use numbered steps for processes\n"
        "✅ Use bold headers (**Header**) only when the answer has multiple distinct sections\n"
        "✅ Keep answers concise when context is limited\n"
        "✅ Write naturally and in a friendly, professional tone\n"
        "✅ Add blank lines between sections for readability\n"
        "\n"
        "STRICTLY FORBIDDEN:\n"
        "❌ NEVER invent information not present in the document\n"
        "❌ NEVER use filler phrases like \"Great question!\" or \"Certainly!\"\n"
        "❌ Write naturally — not like a template\n"
        "\n"
        "═══════════════════════════════════════════════════════════════\n"
        "\n"
        "Context from uploaded documents:\n"
        "{context}\n"
        "\n"
        "STRICT SOURCE RULE:\n"
        "- Your answer must be grounded ONLY in the provided document context above\n"
        "- Do NOT use outside/general knowledge\n"
        "- Do NOT invent information\n"
        "\n"
        "Conversation so far:\n"
        "{chat_history}\n"
        "\n"
        "{language_instruction}\n"
    )

def build_general_prompt(business_context: str) -> str:
    domain = _domain_label(business_context)
    return (
        f"You are an expert AI assistant for: {domain}.\n"
        "\n"
        "⚠️ IMPORTANT: The user's question is NOT covered by the uploaded documents.\n"
        f"You are answering from your GENERAL KNOWLEDGE about {domain}.\n"
        "\n"
        "INSTRUCTIONS:\n"
        "1. Provide helpful, accurate information relevant to this domain\n"
        "2. Be honest if a question is outside your knowledge\n"
        "3. Format your answer clearly and naturally\n"
        "\n"
        "FORMAT RULES:\n"
        "- Use bullet points for lists and options\n"
        "- Use numbered steps for processes or how-tos\n"
        "- Use bold headers only when the answer has multiple distinct sections\n"
        "- Write in a friendly, professional tone\n"
        "- Keep answers focused — do not pad with unrelated information\n"
        "\n"
        "STRICTLY FORBIDDEN:\n"
        "❌ NEVER use filler openers like \"Great question!\" or \"Certainly!\"\n"
        "\n"
        "Conversation so far:\n"
        "{chat_history}\n"
        "\n"
        "{language_instruction}\n"
    )

# ── 6. Query rewriter ─────────────────────────────────────
REWRITE_HUMAN_PROMPT = "Rewrite this question to be explicit: {question}"

if MODEL_PROVIDER == "openai":
    rewrite_llm = ChatOpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )
else:
    rewrite_llm = llm

def rewrite_query(raw_query: str, chat_history_text: str, business_context: str = "") -> str:
    if not chat_history_text.strip():
        return raw_query

    vague_words = [
        "there", "it", "that", "both", "this one", "those",
        "the first one", "second one", "the previous", "same one",
        "the option", "that option", "the item", "that item",
        "here", "same place", "that one"
    ]
    if not any(word in raw_query.lower() for word in vague_words):
        print(f"[QueryRewriter] Already explicit — skipping: '{raw_query}'")
        return raw_query

    print(f"[QueryRewriter] Rewriting vague query: '{raw_query}'")

    system_prompt_text = build_rewrite_prompt(business_context)
    rewrite_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt_text),
        HumanMessagePromptTemplate.from_template(REWRITE_HUMAN_PROMPT)
    ])
    formatted = rewrite_prompt.format_messages(
        chat_history=chat_history_text,
        question=raw_query
    )
    result    = rewrite_llm.invoke(formatted)
    rewritten = result.content.strip() if MODEL_PROVIDER == "openai" else str(result).strip()
    print(f"[QueryRewriter] Result: '{rewritten}'")
    return rewritten


# ── 7. Main answer function ───────────────────────────────
def generate_answer(
    query:            str,
    session_id:       str  = "default",
    use_general:      bool = False,
    language:         str  = "English",
    collection_name:  str  = None,
    business_context: str  = ""
) -> dict:
    """RAG pipeline with language support and domain-aware prompts."""

    memory       = get_memory(session_id)
    history_vars = memory.load_memory_variables({})
    history_msgs = history_vars.get("chat_history", [])

    history_text = ""
    for msg in history_msgs:
        role          = "User" if msg.type == "human" else "Assistant"
        history_text += f"{role}: {msg.content}\n"

    # ── Translate non-English queries to English for search ──
    original_query = query

    # ── Auto-detect the language the user typed in ───────────
    # This overrides the fixed `language` config so the bot
    # always replies in whatever language the user typed,
    # regardless of the widget's data-language setting.
    detected_lang = detect_language(original_query)
    # Only override if detection returned something meaningful
    if detected_lang and detected_lang.lower() != "unknown":
        language = detected_lang

    # ── Translate to English for vector search ───────────────
    if not all(ord(char) < 128 for char in query):
        print(f"[Generator] Non-English query detected: '{query}'")
        print(f"[Generator] Translating to English for search...")
        translate_prompt = f"Translate this to English, keep it concise: {query}"
        translation_response = llm.invoke([
            {"role": "system", "content": "You are a translator. Translate to English."},
            {"role": "user",   "content": translate_prompt}
        ])
        query_for_search = translation_response.content.strip()
        print(f"[Generator] English translation: '{query_for_search}'")
    else:
        query_for_search = query

    # ── Spell-correct the query (domain-aware) ───────────────
    domain_label = _domain_label(business_context)
    spell_prompt = (
        f"Correct spelling mistakes in this query related to {domain_label}.\n"
        "Return ONLY the corrected query.\n"
        "Do not change the meaning.\n"
        f"\nQuery: {query_for_search}"
    )
    spell_response = llm.invoke([
        {"role": "system", "content": f"You correct spelling mistakes in queries related to {domain_label}."},
        {"role": "user",   "content": spell_prompt}
    ])
    query_for_search = spell_response.content.strip()
    print(f"[Generator] Corrected query: '{query_for_search}'")

    # ── Rewrite vague queries ────────────────────────────────
    rewritten_query  = rewrite_query(query_for_search, history_text, business_context)
    lang_instruction = get_language_instruction(language)
    print(f"[Generator] Language: '{language}'")

    # ── Path A: General knowledge ────────────────────────────
    if use_general:
        print(f"[Generator] General knowledge path for: '{original_query}'")

        system_prompt_text = build_general_prompt(business_context)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt_text),
            HumanMessagePromptTemplate.from_template("{question}")
        ])
        formatted = prompt.format_messages(
            chat_history=history_text,
            language_instruction=lang_instruction,
            question=original_query
        )
        result = llm.invoke(formatted)
        answer = result.content.strip() if MODEL_PROVIDER == "openai" else str(result).strip()
        memory.save_context({"input": original_query}, {"answer": answer})
        return {
            "answer":          answer,
            "rewritten_query": rewritten_query,
            "has_pdf_context": False
        }

    # ── Path B: PDF / document retrieval ────────────────────
    docs    = retrieve_docs(rewritten_query, collection_name=collection_name)
    context = "\n".join(docs)

    if not context.strip():
        print(f"[Generator] No document context for: '{original_query}' → asking user")
        return {
            "answer":          None,
            "rewritten_query": rewritten_query,
            "has_pdf_context": False
        }

    print(f"[Generator] Document context found for: '{original_query}'")

    system_prompt_text = build_pdf_prompt(business_context)
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt_text),
        HumanMessagePromptTemplate.from_template("{question}")
    ])
    formatted = prompt.format_messages(
        context=context,
        chat_history=history_text,
        language_instruction=lang_instruction,
        question=original_query
    )
    result = llm.invoke(formatted)
    answer = result.content.strip() if MODEL_PROVIDER == "openai" else str(result).strip()

    if "NO_PDF_CONTEXT" in answer:
        print(f"[Generator] LLM detected irrelevant context → asking user")
        return {
            "answer":          None,
            "rewritten_query": rewritten_query,
            "has_pdf_context": False
        }

    memory.save_context({"input": original_query}, {"answer": answer})
    return {
        "answer":          answer,
        "rewritten_query": rewritten_query,
        "has_pdf_context": True
    }