from pathlib import Path
import sys
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
for p in [str(ROOT_DIR), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from chatbot_engine import chatbot_response

TEST_CASES = [
    ("Greeting", "hai apotekcare", "greeting"),
    ("Operasional", "jam operasional apotek sampai jam berapa?", "jam_operasional"),
    ("Operasional", "apotek buka hari minggu?", "jam_operasional"),
    ("Stok Generic", "stok produk", "cek_stok"),
    ("Stok Batuk", "apakah stok obat batuk tersedia?", "cek_stok"),
    ("Stok Vitamin", "ada vitamin C?", "cek_stok"),
    ("Stok Flu", "apakah ada obat untuk flu?", "cek_stok"),
    ("Rekomendasi", "obat untuk sakit kepala", "rekomendasi_sakit_kepala"),
    ("Rekomendasi", "obat untuk batuk", "rekomendasi_batuk"),
    ("Rekomendasi", "obat untuk maag", "rekomendasi_lambung"),
    ("Rekomendasi", "vitamin untuk daya tahan tubuh", "rekomendasi_vitamin"),
    ("Harga", "berapa harga obat batuk?", "tanya_harga"),
    ("Harga", "berapa harga vitamin C?", "tanya_harga"),
    ("Harga", "harga paracetamol berapa?", "tanya_harga"),
    ("Out of Scope", "siapa presiden indonesia?", "out_of_scope"),
    ("Out of Scope", "berapa harga saham hari ini?", "out_of_scope"),
    ("Medical Boundary", "berapa dosis antibiotik untuk anak?", "medical_boundary"),
]

def main():
    rows = []
    context_products = []
    for category, question, expected_intent in TEST_CASES:
        result = chatbot_response(question, context_products=context_products)
        if result.get("products"):
            context_products = result["products"]
        rows.append({
            "category": category,
            "question": question,
            "expected_intent": expected_intent,
            "actual_intent": result.get("intent"),
            "status": result.get("status"),
            "passed": result.get("intent") == expected_intent,
            "answer_preview": str(result.get("answer", ""))[:180],
        })
    df = pd.DataFrame(rows)
    output_dir = ROOT_DIR / "reports"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "chatbot_scenario_test_results.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("=" * 100)
    print("APOTEKCARE CHATBOT SCENARIO TEST")
    print("=" * 100)
    print(df[["category", "question", "expected_intent", "actual_intent", "status", "passed"]].to_string(index=False))
    print("-" * 100)
    print(f"Passed: {df['passed'].sum()} / {len(df)}")
    print(f"Report saved to: {output_path}")

if __name__ == "__main__":
    main()
