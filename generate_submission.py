"""
generate_submission.py

Generates submission.jsonl — answers for all 30 canonical test pairs.
Run BEFORE deploying to have your submission.jsonl ready.

Usage:
    GROQ_API_KEY=your_key python generate_submission.py

Reads dataset from ./dataset/ (expanded dataset must exist).
Writes ./submission.jsonl  (30 lines, one per test pair).
"""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from composer import compose_message

DATASET_DIR = Path(__file__).parent / "dataset"


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def find_file(directory: Path, name: str) -> Path | None:
    p = directory / f"{name}.json"
    return p if p.exists() else None


def main():
    print("Loading dataset...")

    # Load test pairs
    test_pairs_path = DATASET_DIR / "test_pairs.json"
    if not test_pairs_path.exists():
        print(f"ERROR: {test_pairs_path} not found. Run: python dataset/generate_dataset.py")
        sys.exit(1)

    test_pairs = load_json(test_pairs_path)["pairs"]
    print(f"Found {len(test_pairs)} test pairs")

    # Preload categories
    categories = {}
    cat_dir = DATASET_DIR / "categories"
    for f in cat_dir.glob("*.json"):
        cat = load_json(f)
        categories[cat["slug"]] = cat
    print(f"Loaded {len(categories)} categories: {list(categories.keys())}")

    # Process each test pair
    results = []
    failed = []

    for i, pair in enumerate(test_pairs):
        test_id = pair["test_id"]
        trigger_id = pair["trigger_id"]
        merchant_id = pair["merchant_id"]
        customer_id = pair.get("customer_id")

        print(f"\n[{i+1}/30] {test_id}: merchant={merchant_id} trigger={trigger_id}")

        # Load merchant
        merchant_path = find_file(DATASET_DIR / "merchants", merchant_id)
        if not merchant_path:
            print(f"  SKIP: merchant file not found")
            failed.append(test_id)
            continue
        merchant = load_json(merchant_path)

        # Load category
        cat_slug = merchant.get("category_slug", "")
        category = categories.get(cat_slug)
        if not category:
            print(f"  SKIP: category '{cat_slug}' not found")
            failed.append(test_id)
            continue

        # Load trigger
        trigger_path = find_file(DATASET_DIR / "triggers", trigger_id)
        if not trigger_path:
            print(f"  SKIP: trigger file not found")
            failed.append(test_id)
            continue
        trigger = load_json(trigger_path)

        # Load customer (optional)
        customer = None
        if customer_id:
            customer_path = find_file(DATASET_DIR / "customers", customer_id)
            if customer_path:
                customer = load_json(customer_path)

        # Compose
        try:
            result = compose_message(
                category=category,
                merchant=merchant,
                trigger=trigger,
                customer=customer,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(test_id)
            continue

        if not result.get("body"):
            print(f"  ERROR: empty body")
            failed.append(test_id)
            continue

        # Build submission line
        send_as = "merchant_on_behalf" if customer_id else "vera"
        line = {
            "test_id": test_id,
            "merchant_id": merchant_id,
            "trigger_id": trigger_id,
            "customer_id": customer_id,
            "body": result["body"],
            "cta": result.get("cta", "open_ended"),
            "send_as": send_as,
            "suppression_key": trigger.get("suppression_key", ""),
            "rationale": result.get("rationale", ""),
        }

        results.append(line)
        print(f"  OK  body={result['body'][:80]}...")
        print(f"      cta={result.get('cta')} | rationale={result.get('rationale','')[:60]}...")

    # Write output
    out_path = Path(__file__).parent / "submission.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Done! {len(results)}/30 pairs generated → {out_path}")
    if failed:
        print(f"Failed: {failed}")

    # Quick quality check
    print("\nQuality spot-check (first 3 entries):")
    for r in results[:3]:
        print(f"\n  {r['test_id']}: {r['merchant_id']}")
        print(f"  BODY: {r['body']}")
        print(f"  CTA: {r['cta']} | RATIONALE: {r['rationale']}")


if __name__ == "__main__":
    main()
