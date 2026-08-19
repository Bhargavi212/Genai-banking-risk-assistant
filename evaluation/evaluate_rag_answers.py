import re
import time
from pathlib import Path

import pandas as pd

from Application.services import rag_engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "docs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


TEST_CASES = [
    {
        "question": "What is Know Your Customer and why is it important?",
        "expected_source": "Know-Your-Customer-White-Paper-2022-compressed.pdf",
        "expect_refusal": False,
    },
    {
        "question": "Why do financial institutions perform customer due diligence?",
        "expected_source": "Know-Your-Customer-White-Paper-2022-compressed.pdf",
        "expect_refusal": False,
    },
    {
        "question": "What information can be used to verify a customer's identity?",
        "expected_source": None,
        "expect_refusal": True,
    },
    {
        "question": "What should a banking regulatory compliance program include?",
        "expected_source": "Banking Regulatory Compliance Checklist.pdf",
        "expect_refusal": False,
    },
    {
        "question": "How can compliance monitoring help identify regulatory weaknesses?",
        "expected_source": "Banking Regulatory Compliance Checklist.pdf",
        "expect_refusal": False,
    },
]


REFUSAL_TEXT = (
    "The retrieved documents do not provide enough information "
    "to answer this question."
)


def extract_citations(answer):
    pattern = r"\[Source:\s*(.*?)\s*\|\s*Page:\s*(\d+)\]"
    return re.findall(pattern, answer)


def evaluate_answer(case, answer):
    citations = extract_citations(answer)

    has_citation = len(citations) > 0

    cited_sources = [
        filename.strip()
        for filename, _ in citations
    ]

    cited_pages = [
        int(page)
        for _, page in citations
    ]

    refused = REFUSAL_TEXT.lower() in answer.lower()

    if case["expect_refusal"]:
        refusal_correct = refused
        expected_source_present = True
    else:
        refusal_correct = not refused
        expected_source_present = (
            case["expected_source"] in cited_sources
        )

    valid_page_citation = all(
        page > 0
        for page in cited_pages
    ) if citations else False

    if case["expect_refusal"]:
        citation_behavior_correct = not has_citation
    else:
        citation_behavior_correct = (
            has_citation
            and expected_source_present
            and valid_page_citation
        )

    return {
        "has_citation": has_citation,
        "expected_source_present": expected_source_present,
        "valid_page_citation": valid_page_citation,
        "refused": refused,
        "refusal_correct": refusal_correct,
        "citation_behavior_correct": citation_behavior_correct,
    }


def main():
    rows = []

    for case in TEST_CASES:
        question = case["question"]

        print("\n" + "=" * 80)
        print("Question:", question)

        context = rag_engine.retrieve_context(
            question,
            rag_engine.index,
            rag_engine.chunks,
            rag_engine.sources,
            k=3,
        )

        start = time.perf_counter()

        try:
            answer = rag_engine.query_compliance(
                question,
                context,
            )

            error = ""

        except Exception as exc:
            answer = ""
            error = str(exc)

        latency = time.perf_counter() - start

        if error:
            checks = {
                "has_citation": False,
                "expected_source_present": False,
                "valid_page_citation": False,
                "refused": False,
                "refusal_correct": False,
                "citation_behavior_correct": False,
            }
        else:
            checks = evaluate_answer(
                case,
                answer,
            )

        rows.append(
            {
                "question": question,
                "expected_source": case["expected_source"],
                "expect_refusal": case["expect_refusal"],
                "answer": answer,
                "generation_latency_seconds": round(
                    latency,
                    4,
                ),
                "error": error,
                **checks,
            }
        )

        print("\nAnswer:")
        print(answer if answer else error)

        print(
            "\nCitation behavior:",
            checks["citation_behavior_correct"],
        )

        print(
            "Refusal behavior:",
            checks["refusal_correct"],
        )

    results = pd.DataFrame(rows)

    successful = results["error"] == ""

    evaluated = results[successful]

    print("\n\nRAG Answer Evaluation")
    print("---------------------")
    print(f"Questions: {len(results)}")

    if len(evaluated):
        print(
            "Citation behavior accuracy:",
            f"{evaluated['citation_behavior_correct'].mean():.4f}",
        )

        print(
            "Refusal behavior accuracy:",
            f"{evaluated['refusal_correct'].mean():.4f}",
        )

        print(
            "Average generation latency:",
            f"{evaluated['generation_latency_seconds'].mean():.4f}",
            "seconds",
        )

    print(
        "API errors:",
        int((~successful).sum()),
    )

    output_path = (
        RESULTS_DIR
        / "rag_answer_evaluation.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print("\nResults saved to:")
    print(output_path)


if __name__ == "__main__":
    main()
