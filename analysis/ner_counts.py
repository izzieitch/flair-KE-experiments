import json
from collections import Counter

NER_RESULTS_FILE = "results/flair_ner_results.json"


def main():
    with open(NER_RESULTS_FILE, "r", encoding="utf-8") as f:
        ner_results = json.load(f)

    label_counter = Counter()

    # Each item in ner_results is a sentence record with an "entities" list
    for record in ner_results:
        for ent in record.get("entities", []):
            label = ent.get("label")
            if label:
                label_counter[label] += 1

    print("\nEntity counts per label:\n")
    for label, count in label_counter.most_common():
        print(f"{label:15s} {count}")

    print(f"\nTotal entities recognised: {sum(label_counter.values())}")
    print(f"Total sentences processed: {len(ner_results)}")


if __name__ == "__main__":
    main()