import json
from collections import Counter
import matplotlib.pyplot as plt

INPUT_JSON = "results/hybrid_spacy_on_flair.json"

MAPPED_RELATIONS = {
    "ESTABLISHED_BY",
    "ESTABLISHED_IN",
    "MOVED_TO",
    "LOADED_BY",
    "AFFECTED_BY",
    "BECAME"
}


def main():
    # Load data
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_counter = Counter()
    mapped_counter = Counter()
    open_counter = Counter()

    # Count relations
    for row in data:
        rel = row.get("relation")

        if not rel:
            continue

        total_counter[rel] += 1

        if rel in MAPPED_RELATIONS:
            mapped_counter[rel] += 1
        else:
            open_counter[rel] += 1

    # Print summary
    print("\n=== Summary ===")
    print(f"Total relations: {sum(total_counter.values())}")
    print(f"Mapped relations: {sum(mapped_counter.values())}")
    print(f"Open relations: {sum(open_counter.values())}")

    # Top N
    top_n = 15

    # --- Plot 1: All relations ---
    plot_counter(total_counter, top_n, "Top Relations (All)")

    # --- Plot 2: Mapped relations ---
    if mapped_counter:
        plot_counter(mapped_counter, top_n, "Mapped (Rule-based) Relations")

    # --- Plot 3: Open relations ---
    plot_counter(open_counter, top_n, "Open (Verb-based) Relations")


def plot_counter(counter, top_n, title):
    most_common = counter.most_common(top_n)

    labels = [x[0] for x in most_common]
    values = [x[1] for x in most_common]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)

    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Relation")
    plt.ylabel("Frequency")
    plt.title(title)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
