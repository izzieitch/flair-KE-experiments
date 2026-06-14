import os
import json

# ---------- config ----------
FLAIR_INPUT_FILE = "results/flair_ner_results.json"
OUTPUT_FILE = "results/flair_ner_with_manual.json"
BASE_FOLDER = "opium_sentences_complete"
# ----------------------------------

# ---------- load manual matches (fixed) ----------
def load_manual_matches(base_folder):
    manual_by_sentence = {}

    TARGET_TERMS = {"opium", "opiate", "opiates", "poppy", "poppies"}

    for root, dirs, files in os.walk(base_folder):
        match_type = os.path.basename(root)

        if match_type not in {"exact", "fuzzy"}:
            continue

        concept = os.path.basename(os.path.dirname(root))
        label_type = "EXACT" if match_type == "exact" else "FUZZY"

        for filename in files:
            if not filename.endswith(".json"):
                continue

            path = os.path.join(root, filename)
            if os.path.getsize(path) == 0:
                continue

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for sentence in data.get("sentences", []):
                sent_id = sentence.get("sentence_id")
                text = sentence.get("text", "")

                # ✅ FIX HERE
                if match_type == "exact":
                    tokens = [
                        t for t in sentence.get("tokens", [])
                        if t.lower() in TARGET_TERMS
                    ]
                else:
                    tokens = sentence.get("fuzzy_tokens", [])

                for token in tokens:
                    start = text.lower().find(token.lower())
                    if start == -1:
                        continue

                    end = start + len(token)

                    manual_by_sentence.setdefault(sent_id, []).append({
                        "text": token,
                        "label": f"MANUAL_{concept.upper()}_{label_type}",
                        "start_pos": start,
                        "end_pos": end,
                        "score": 1.0
                    })

    return manual_by_sentence


# ---------- overlap helper ----------
def overlaps(a, b):
    return not (a["end_pos"] <= b["start_pos"] or a["start_pos"] >= b["end_pos"])


# ---------- deduplicate helper ----------
def deduplicate(entities):
    seen = set()
    result = []

    for e in entities:
        key = (e["start_pos"], e["end_pos"], e["label"])
        if key not in seen:
            seen.add(key)
            result.append(e)

    return result


# ---------- main ----------
def main():
    # load flair output
    with open(FLAIR_INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    manual_by_sentence = load_manual_matches(BASE_FOLDER)

    updated_results = []

    for item in data:
        sent_id = item.get("sent_id")
        existing_entities = item.get("entities", [])

        manual_entities = manual_by_sentence.get(sent_id, [])

        # sort: exact first, fuzzy after
        manual_entities = sorted(
            manual_entities,
            key=lambda x: (x["start_pos"], "FUZZY" in x["label"])
        )

        filtered_manual = []

        for m in manual_entities:
            # skip overlap with Flair entities
            if any(overlaps(m, e) for e in existing_entities):
                continue

            # skip overlap with already accepted manual entities
            if any(overlaps(m, fm) for fm in filtered_manual):
                continue

            filtered_manual.append(m)

        # merge + deduplicate
        item["entities"].extend(filtered_manual)
        item["entities"] = deduplicate(item["entities"])

        updated_results.append(item)

    # save output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(updated_results, f, indent=2, ensure_ascii=False)

    print("Injection complete.")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
