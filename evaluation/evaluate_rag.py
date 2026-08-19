from pathlib import Path
import time

import pandas as pd

from Application.services import rag_engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "docs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


TEST_CASES = [
    {
        "question": "What is the purpose of customer due diligence in KYC?",
        "expected_keyword": "customer",
    },
    {
        "question": "What factors are considered in a BSA/AML risk assessment?",
        "expected_keyword": "risk",
    },
    {
        "question": "Why is transaction monitoring important for AML compliance?",
        "expected_keyword": "transaction",
    },
]


def evaluate_retrieval():
    results = []

    for case in TEST_CASES:
        start = time.perf_counter()

        context = rag_engine.retrieve_context(
            case["question"],
            rag_engine.index,
            rag_engine.chunks,
            rag_engine.sources,
            k=3,
        )

        latency = time.perf_counter() - start

        keyword_found = (
            case["expected_keyword"].lower()
            in context.lower()
        )

        results.append(
            {
                "question": case["question"],
                "expected_keyword": case["expected_keyword"],
                "retrieval_hit": keyword_found,
                "latency_seconds": round(latency, 4),
            }
        )

    return pd.DataFrame(results)


def main():
    df = evaluate_retrieval()

    hit_rate = df["retrieval_hit"].mean()
    avg_latency = df["latency_seconds"].mean()

    print("\nRAG Retrieval Evaluation")
    print("------------------------")
    print(f"Questions evaluated: {len(df)}")
    print(f"Retrieval hit rate: {hit_rate:.4f}")
    print(f"Average latency: {avg_latency:.4f} seconds")

    csv_path = RESULTS_DIR / "rag_retrieval_results.csv"
    df.to_csv(csv_path, index=False)

    md_path = RESULTS_DIR / "rag_evaluation.md"

    with md_path.open("w", encoding="utf-8") as file:
        file.write("# RAG Retrieval Evaluation\n\n")
        file.write(
            "This evaluation measures whether retrieved "
            "document context contains expected evidence "
            "for a small set of compliance questions.\n\n"
        )

        file.write(f"- Questions evaluated: {len(df)}\n")
        file.write(
            f"- Retrieval hit rate: {hit_rate:.4f}\n"
        )
        file.write(
            f"- Average retrieval latency: "
            f"{avg_latency:.4f} seconds\n\n"
        )

        file.write(
            "This is an initial prototype evaluation and "
            "does not yet represent a comprehensive "
            "groundedness or faithfulness benchmark.\n"
        )

    print(f"\nSaved results to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
