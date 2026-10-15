"""Evaluation runner benchmarking the 12 labeled test questions."""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add repository root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.core.config import get_settings
from backend.app.graph.builder import build_rag_graph
from backend.app.graph.state import create_initial_state
from backend.app.services.database import SQLiteDatabaseService
from backend.app.services.embeddings import SentenceTransformerEmbeddingsService
from backend.app.services.llm import GroqLLMService, MockLLMService
from backend.app.services.vector_store import ChromaVectorStoreService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_runner")


async def evaluate_suite(
    live_mode: bool = False,
    output_path: Path = Path(__file__).resolve().parent / "evaluation_results.json",
) -> None:
    settings = get_settings()
    questions_file = Path(__file__).resolve().parent / "questions.json"

    if not questions_file.exists():
        logger.error("Questions file not found: %s", questions_file)
        sys.exit(1)

    questions = json.loads(questions_file.read_text(encoding="utf-8"))

    # Determine mode
    has_key = bool(settings.groq_api_key and settings.groq_api_key.startswith("gsk_"))
    is_live = live_mode and has_key

    if live_mode and not has_key:
        logger.warning(
            "Live evaluation requested but valid GROQ_API_KEY is not set. "
            "Running in OFFLINE MOCK MODE. Do not claim this ran with live credentials."
        )
        is_live = False

    logger.info("Initializing services for evaluation (is_live=%s)...", is_live)
    embeddings = SentenceTransformerEmbeddingsService(settings.embedding_model)
    vector_store = ChromaVectorStoreService(settings.chroma_dir, settings.chroma_collection)

    if is_live:
        llm = GroqLLMService(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    else:
        llm = MockLLMService()

    graph = build_rag_graph(
        llm=llm,
        embeddings=embeddings,
        vector_store=vector_store,
        top_k=settings.top_k,
        grading_concurrency=settings.grading_concurrency,
    )

    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 90)
    print(f"RUNNING RAG BENCHMARK EVALUATION ({'LIVE GROQ: ' + settings.llm_model if is_live else 'OFFLINE MOCK MODE'})")
    print("=" * 90)

    for item in questions:
        qid = item["id"]
        q_text = item["question"]
        cat = item["category"]
        exp_status = item["expected_status"]

        logger.info("Evaluating [%s] (%s): %s", qid, cat, q_text[:60])

        start_time = time.perf_counter()
        state = create_initial_state(q_text, max_retries=settings.max_retries)

        try:
            final_state = await graph.ainvoke(state)
            elapsed_sec = round(time.perf_counter() - start_time, 3)

            actual_status = final_state["status"]
            citations = final_state.get("citations", [])
            retrieval_attempts = final_state["retrieval_attempts"]
            retry_count = final_state["retry_count"]

            status_match = (actual_status == exp_status)

            # Check citation validity
            citation_valid = True
            for c in citations:
                if not c.get("chunk_id") or not c.get("source"):
                    citation_valid = False

            rec = {
                "id": qid,
                "category": cat,
                "question": q_text,
                "expected_status": exp_status,
                "actual_status": actual_status,
                "status_match": status_match,
                "retrieval_attempts": retrieval_attempts,
                "retry_count": retry_count,
                "citation_count": len(citations),
                "citation_valid": citation_valid,
                "latency_sec": elapsed_sec,
                "answer_preview": final_state["answer"][:120] + "..." if len(final_state["answer"]) > 120 else final_state["answer"],
            }
            results.append(rec)

        except Exception as e:
            elapsed_sec = round(time.perf_counter() - start_time, 3)
            logger.error("Error evaluating %s: %s", qid, e)
            results.append({
                "id": qid,
                "category": cat,
                "question": q_text,
                "expected_status": exp_status,
                "actual_status": "error",
                "status_match": False,
                "retrieval_attempts": 0,
                "retry_count": 0,
                "citation_count": 0,
                "citation_valid": False,
                "latency_sec": elapsed_sec,
                "error": str(e),
            })

    # Summary calculations
    total = len(results)
    matched = sum(1 for r in results if r.get("status_match"))
    avg_latency = round(sum(r.get("latency_sec", 0) for r in results) / total, 3) if total else 0.0

    print("\n" + "-" * 90)
    print(f"{'ID':<10} | {'CATEGORY':<20} | {'STATUS':<12} | {'MATCH':<6} | {'ATTEMPTS':<8} | {'LATENCY':<8}")
    print("-" * 90)
    for r in results:
        status_disp = r.get("actual_status", "error")
        match_disp = "PASS" if r.get("status_match") else "FAIL"
        print(f"{r['id']:<10} | {r['category']:<20} | {status_disp:<12} | {match_disp:<6} | {r['retrieval_attempts']:<8} | {r['latency_sec']}s")
    print("-" * 90)
    print(f"Total Questions: {total} | Accuracy: {matched}/{total} ({round(matched/total*100, 1)}%) | Avg Latency: {avg_latency}s")
    print("=" * 90 + "\n")

    report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "live_groq" if is_live else "offline_mock",
        "model": settings.llm_model if is_live else "MockLLMService",
        "embedding_model": settings.embedding_model,
        "total_questions": total,
        "status_accuracy": round(matched / total, 3) if total else 0.0,
        "average_latency_sec": avg_latency,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Evaluation results saved to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation benchmark over 12 labeled questions.")
    parser.add_argument("--live", action="store_true", help="Run against live Groq model using GROQ_API_KEY.")
    args = parser.parse_args()

    asyncio.run(evaluate_suite(live_mode=args.live))


if __name__ == "__main__":
    main()
