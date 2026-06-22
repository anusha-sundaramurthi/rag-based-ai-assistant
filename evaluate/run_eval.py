
import sys
import os
import asyncio
import time
import json

# ── Make sure src/ is importable from project root ───────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,
    ContextRecall,
)

from src.retriever import retrieve_docs
from src.generator import generate_answer, clear_memory
from evaluate.test_dataset import test_cases


# ── 1. RAGAS judge LLM — Groq (free) ─────────────────────
# RAGAS uses this LLM internally to evaluate answers
ragas_llm = LangchainLLMWrapper(ChatOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model_name="llama-3.3-70b-versatile",
    temperature=0
))

# ── 2. RAGAS embeddings — OpenAI (needed for relevancy) ──
# ResponseRelevancy metric needs embeddings internally
ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=os.getenv("OPENAI_API_KEY")
))


# ── 3. Per-metric async scorers (from notebook pattern) ───

async def score_faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    """Is the answer grounded in the retrieved context? No hallucination?"""
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts
    )
    scorer = Faithfulness(llm=ragas_llm)
    return await scorer.single_turn_ascore(sample)


async def score_response_relevancy(question: str, answer: str, contexts: list[str]) -> float:
    """Does the answer actually address the question asked?"""
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts
    )
    scorer = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    return await scorer.single_turn_ascore(sample)


async def score_context_precision(question: str, answer: str, contexts: list[str], ground_truth: str) -> float:
    """Are the retrieved chunks relevant to the question?"""
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=ground_truth
    )
    scorer = ContextPrecision(llm=ragas_llm)
    return await scorer.single_turn_ascore(sample)


async def score_context_recall(question: str, answer: str, contexts: list[str], ground_truth: str) -> float:
    """Did retrieval fetch all the information needed to answer?"""
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=ground_truth
    )
    scorer = ContextRecall(llm=ragas_llm)
    return await scorer.single_turn_ascore(sample)


async def score_all_metrics(question: str, answer: str, contexts: list[str], ground_truth: str) -> dict:
    """Run all 4 metrics concurrently for one test case."""
    results = await asyncio.gather(
        score_faithfulness(question, answer, contexts),
        score_response_relevancy(question, answer, contexts),
        score_context_precision(question, answer, contexts, ground_truth),
        score_context_recall(question, answer, contexts, ground_truth),
        return_exceptions=True   # don't crash if one metric fails
    )

    def safe(val, name):
        if isinstance(val, Exception):
            print(f"    ⚠️  {name} failed: {val}")
            return None
        return round(float(val), 4)

    return {
        "faithfulness":        safe(results[0], "faithfulness"),
        "response_relevancy":  safe(results[1], "response_relevancy"),
        "context_precision":   safe(results[2], "context_precision"),
        "context_recall":      safe(results[3], "context_recall"),
    }


# ── 4. Collect answers from your RAG pipeline ─────────────

def collect_rag_outputs() -> list[dict]:
    """
    Run each test question through your actual RAG pipeline
    and collect question, answer, contexts, ground_truth.
    """
    print("\n" + "="*60)
    print("STEP 1 — Collecting RAG pipeline outputs")
    print("="*60)

    collected = []

    for i, case in enumerate(test_cases):
        question     = case["question"]
        ground_truth = case["ground_truth"]

        print(f"\n[{i+1}/{len(test_cases)}] {question[:60]}...")

        # Get retrieved chunks directly
        contexts = retrieve_docs(question)

        if not contexts:
            print(f"  ⚠️  No context retrieved — skipping")
            continue

        # Get generated answer from your pipeline
        result = generate_answer(
            query=question,
            session_id=f"eval_session_{i}",
            use_general=False,
            language="English"
        )

        answer = result.get("answer")

        if answer is None:
            print(f"  ⚠️  No answer generated (LLM returned NO_PDF_CONTEXT) — skipping")
            # Clear session memory before next question
            clear_memory(f"eval_session_{i}")
            continue

        print(f"  ✅ Answer: {answer[:80]}...")
        print(f"  📄 Contexts: {len(contexts)} chunks")

        collected.append({
            "question":     question,
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": ground_truth,
        })

        # Clear session memory so questions don't bleed into each other
        clear_memory(f"eval_session_{i}")

        # Small delay to avoid Groq rate limits during collection
        time.sleep(1)

    print(f"\n✅ Collected {len(collected)}/{len(test_cases)} test cases")
    return collected


# ── 5. Run RAGAS evaluation ───────────────────────────────

async def run_evaluation(collected: list[dict]) -> list[dict]:
    """
    Evaluate each collected result using RAGAS SingleTurnSample.
    Adds a delay between cases to avoid Groq rate limits.
    """
    print("\n" + "="*60)
    print("STEP 2 — Running RAGAS evaluation")
    print("="*60)

    all_results = []

    for i, item in enumerate(collected):
        print(f"\n[{i+1}/{len(collected)}] Evaluating: {item['question'][:55]}...")

        scores = await score_all_metrics(
            question=item["question"],
            answer=item["answer"],
            contexts=item["contexts"],
            ground_truth=item["ground_truth"],
        )

        result = {**item, **scores}
        all_results.append(result)

        print(f"  faithfulness:       {scores['faithfulness']}")
        print(f"  response_relevancy: {scores['response_relevancy']}")
        print(f"  context_precision:  {scores['context_precision']}")
        print(f"  context_recall:     {scores['context_recall']}")

        # Delay between evaluations to avoid Groq TPM rate limit
        # RAGAS makes ~5-8 LLM calls per metric internally
        if i < len(collected) - 1:
            print(f"  ⏳ Waiting 3s before next evaluation...")
            await asyncio.sleep(3)

    return all_results


# ── 6. Print summary and save results ────────────────────

def print_summary(results: list[dict]):
    metrics = ["faithfulness", "response_relevancy", "context_precision", "context_recall"]

    print("\n" + "="*60)
    print("RAGAS EVALUATION RESULTS SUMMARY")
    print("="*60)

    averages = {}
    for metric in metrics:
        scores = [r[metric] for r in results if r[metric] is not None]
        avg = round(sum(scores) / len(scores), 4) if scores else 0.0
        averages[metric] = avg
        bar = "█" * int(avg * 20)
        print(f"  {metric:<22} {avg:.4f}  {bar}")

    print("\n  What the scores mean:")
    print("  > 0.80 = Good   |  0.60–0.80 = Acceptable   |  < 0.60 = Needs work")

    return averages


def save_results(results: list[dict], averages: dict):
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # Save per-question JSON
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump({
            "averages": averages,
            "per_question": results
        }, f, indent=2)
    print(f"\n✅ Detailed results saved → evaluate/results.json")

    # Save CSV for easy viewing in Excel/Sheets
    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, "w") as f:
        headers = ["question", "faithfulness", "response_relevancy",
                   "context_precision", "context_recall", "answer"]
        f.write(",".join(headers) + "\n")
        for r in results:
            row = [
                f'"{r["question"]}"',
                str(r["faithfulness"]),
                str(r["response_relevancy"]),
                str(r["context_precision"]),
                str(r["context_recall"]),
                f'"{r["answer"][:100].replace(chr(34), "")}"'
            ]
            f.write(",".join(row) + "\n")
    print(f"✅ CSV results saved     → evaluate/results.csv")


# ── 7. Main ───────────────────────────────────────────────

async def main():
    print("\n🧭 AI Travel Assistant — RAGAS Evaluation")
    print("   LLM judge : Groq llama-3.3-70b (free)")
    print("   Embeddings: OpenAI text-embedding-3-small")
    print("   Test cases:", len(test_cases))

    # Step 1 — collect RAG outputs
    collected = collect_rag_outputs()

    if not collected:
        print("\n❌ No test cases collected. Make sure:")
        print("   1. Your PDF is uploaded to Qdrant")
        print("   2. QDRANT_HOST, QDRANT_API_KEY, COLLECTION_NAME are set in .env")
        print("   3. OPENAI_API_KEY and GROQ_API_KEY are set in .env")
        return

    # Step 2 — evaluate with RAGAS
    results = await run_evaluation(collected)

    # Step 3 — print summary
    averages = print_summary(results)

    # Step 4 — save results
    save_results(results, averages)

    print("\n🎉 Evaluation complete!")


if __name__ == "__main__":
    asyncio.run(main())
