import json
from collections import Counter, defaultdict

INPUT_JSON = "results/hybrid_spacy_on_flair.json"

def analyze_relations():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    relation_counts = Counter()
    mapped_relations = set([
        "ESTABLISHED_BY",
        "ESTABLISHED_IN",
        "MOVED_TO",
        "LOADED_BY",
        "AFFECTED_BY",
        "BECAME"
    ])

    mapped_counter = Counter()
    open_counter = Counter()

    verb_counter = Counter()
    dep_patterns = Counter()

    for row in data:
        rel = row["relation"]
        verb = row.get("verb_lemma", "UNKNOWN")
        dep_pattern = (row.get("dep_subject"), row.get("dep_object"))

        relation_counts[rel] += 1
        verb_counter[verb] += 1
        dep_patterns[dep_pattern] += 1

        if rel in mapped_relations:
            mapped_counter[rel] += 1
        else:
            open_counter[rel] += 1

    print("\n=== Top Relations ===")
    for rel, count in relation_counts.most_common(20):
        print(f"{rel}: {count}")

    print("\n=== Mapped (Rule-based) Relations ===")
    for rel, count in mapped_counter.most_common():
        print(f"{rel}: {count}")

    print("\n=== Open (Verb-based) Relations ===")
    for rel, count in open_counter.most_common(20):
        print(f"{rel}: {count}")

    print("\n=== Top Verbs ===")
    for verb, count in verb_counter.most_common(20):
        print(f"{verb}: {count}")

    print("\n=== Top Dependency Patterns ===")
    for deps, count in dep_patterns.most_common(20):
        print(f"{deps}: {count}")

    # Optional: Save summary
    summary = {
        "total_relations": len(data),
        "unique_relations": len(relation_counts),
        "mapped_relations": dict(mapped_counter),
        "open_relations": dict(open_counter),
        "top_relations": relation_counts.most_common(20),
    }

    with open("results/verb_relation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary saved to results/verb_relation_summary.json")


if __name__ == "__main__":
    analyze_relations()
