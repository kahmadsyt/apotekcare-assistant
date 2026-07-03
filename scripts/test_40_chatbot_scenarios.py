"""
Uji 40 skenario ApotekCare Assistant:
- 20 pertanyaan in-scope yang seharusnya dapat dijawab
- 20 pertanyaan out-of-scope yang harus ditolak/fallback domain

Jalankan dari root project:
    python scripts/test_40_chatbot_scenarios.py

Output:
    reports/chatbot_40_scenario_test_results.csv
"""

from pathlib import Path
import sys
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
TEST_PATH = ROOT_DIR / "data" / "tests" / "chatbot_40_test_questions.csv"
OUTPUT_DIR = ROOT_DIR / "reports"
OUTPUT_PATH = OUTPUT_DIR / "chatbot_40_scenario_test_results.csv"

for path in [str(ROOT_DIR), str(APP_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from chatbot_engine import chatbot_response  # noqa: E402


def normalize_intent(intent: str) -> str:
    return str(intent).strip().lower()


def is_passed(actual_intent: str, expected_intent: str, scope: str) -> bool:
    actual = normalize_intent(actual_intent)
    expected = normalize_intent(expected_intent)

    if actual == expected:
        return True

    # Toleransi untuk pertanyaan in-scope yang dijawab benar melalui intent stok/rekomendasi/harga.
    if scope == "in_scope":
        if expected == "cek_stok" and actual in {"cek_stok", "rekomendasi_batuk", "rekomendasi_flu", "rekomendasi_vitamin"}:
            return True
        if expected == "rekomendasi_lambung" and actual in {"rekomendasi_lambung", "rekomendasi_maag"}:
            return True

    return False


def main():
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"File test tidak ditemukan: {TEST_PATH}")

    tests = pd.read_csv(TEST_PATH)

    rows = []
    context_products = []

    # Pertanyaan "harganya berapa?" butuh konteks, maka context_products disimpan dari jawaban sebelumnya.
    for _, row in tests.iterrows():
        question = row["question"]
        result = chatbot_response(question, context_products=context_products)

        if result.get("products"):
            context_products = result["products"]

        actual_intent = result.get("intent", "")
        passed = is_passed(actual_intent, row["expected_intent"], row["scope"])

        rows.append(
            {
                "case_id": row["case_id"],
                "scope": row["scope"],
                "category": row["category"],
                "question": question,
                "expected_intent": row["expected_intent"],
                "actual_intent": actual_intent,
                "status": result.get("status", ""),
                "passed": passed,
                "answer_preview": str(result.get("answer", ""))[:250].replace("\n", " "),
            }
        )

    results = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    passed_count = int(results["passed"].sum())
    total_count = len(results)
    accuracy = passed_count / total_count if total_count else 0

    print("=" * 100)
    print("APOTEKCARE ASSISTANT - 40 SCENARIO TEST")
    print("=" * 100)
    print(results[["case_id", "scope", "category", "question", "expected_intent", "actual_intent", "status", "passed"]].to_string(index=False))
    print("-" * 100)
    print(f"PASSED      : {passed_count}/{total_count}")
    print(f"PASS RATE   : {accuracy:.2%}")
    print(f"OUTPUT FILE : {OUTPUT_PATH}")
    print("=" * 100)

    failed = results[~results["passed"]]
    if not failed.empty:
        print("\nFAILED CASES:")
        print(failed[["case_id", "question", "expected_intent", "actual_intent", "status", "answer_preview"]].to_string(index=False))


if __name__ == "__main__":
    main()
