from pathlib import Path
import time

import numpy as np
import pandas as pd

from Application.services import rag_engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "docs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Small manually labeled evaluation set.
# Each question is paired with the document expected to contain the answer.
TEST_CASES = [
    {
        "question": "What is the purpose of a BSA AML risk assessment?",
        "expected_source": "BSA_AML Risk Assessment.pdf",
    },
    {
        "question": "What factors should banks consider when assessing AML risk?",
        "expected_source": "BSA_AML Risk Assessment.pdf",
    },
    {
        "question": "What is Know Your Customer and why is it important?",
        "expected_source": "Know-Your-Customer-White-Paper-2022-compressed.pdf",
    },
    {
        "question": "Why do financial institutions perform customer due diligence?",
        "expected_source": "Know-Your-Customer-White-Paper-2022-compressed.pdf",
    },
    {
        "question": "What should a banking regulatory compliance program include?",
        "expected_source": "Banking Regulatory Compliance Checklist.pdf",
    },
]


def retrieve_sources(question, k):
    question_vector = rag_engine.embed_model.encode([question])

    _, indices = rag_engine.index.search(
        np.asarray(question_vector, dtype="float32"),
        k,
    )

    return [
        rag_engine.sources[i]
        for i in indices[0]
        if 0 <= i < len(rag_engine.sources)
    ]


def main():
    rows = []

    for case in TEST_CASES:
        start = time.perf_counter()

        top5_sources = retrieve_sources(
            case["question"],
            k=5,
        )

        latency = time.perf_counter() - start

        expected = case["expected_source"]

        recall_1 = int(
            expected in top5_sources[:1]
        )

        recall_3 = int(
            expected in top5_sources[:3]
        )

        recall_5 = int(
            expected in top5_sources[:5]
        )

        rows.append(
            {
                "question": case["question"],
                "expected_source": expected,
                "top_1_source": (
                    top5_sources[0]
                    if top5_sources
                    else ""
                ),
                "recall@1": recall_1,
                "recall@3": recall_3,
                "recall@5": recall_5,
                "latency_seconds": round(
                    latency,
                    4,
                ),
            }
        )

    results = pd.DataFrame(rows)

    recall_1 = results["recall@1"].mean()
    recall_3 = results["recall@3"].mean()
    recall_5 = results["recall@5"].mean()

    avg_latency = results[
        "latency_seconds"
    ].mean()

    print("\nRAG Retrieval Evaluation")
    print("------------------------")

    print(
        f"Questions: {len(results)}"
    )

    print(
        f"Recall@1: {recall_1:.4f}"
    )

    print(
        f"Recall@3: {recall_3:.4f}"
    )

    print(
        f"Recall@5: {recall_5:.4f}"
    )

    print(
        f"Average latency: "
        f"{avg_latency:.4f} seconds"
    )

    print("\nPer-question results:")
    print(
        results[
            [
                "question",
                "expected_source",
                "top_1_source",
                "recall@1",
                "recall@3",
                "recall@5",
            ]
        ].to_string(index=False)
    )

    results.to_csv(
        RESULTS_DIR
        / "rag_retrieval_results.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
