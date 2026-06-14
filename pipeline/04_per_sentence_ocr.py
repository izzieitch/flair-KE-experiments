import os
import json
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

# ---------- configuration ----------
BASE_FOLDER = "opium_sentences_complete"

EXPANDED_NLTK = "vocab/nltk_words_expanded.txt"
GUTENBERG_EXTRA_VOCAB = "vocab/gutenberg_not_in_nltk_words.txt"
# ----------------------------------


# load vocabularies
with open(EXPANDED_NLTK, "r", encoding="utf-8") as f:
    expanded_nltk = set(line.strip().lower() for line in f if line.strip())

with open(GUTENBERG_EXTRA_VOCAB, "r", encoding="utf-8") as f:
    gutenberg_extra_vocab = set(line.strip().lower() for line in f if line.strip())

combined_vocab = expanded_nltk | gutenberg_extra_vocab


def compute_ocr_quality(text, vocab):
    tokens = word_tokenize(text)
    if not tokens:
        return 0.0
    recognized = sum(1 for t in tokens if t.lower() in vocab)
    return (recognized / len(tokens)) * 100


def main():
    exact_scores = []
    fuzzy_scores = []

    for root, dirs, files in os.walk(BASE_FOLDER):
        match_type = os.path.basename(root)
        if match_type not in {"exact", "fuzzy"}:
            continue

        for filename in files:
            if not filename.endswith(".json"):
                continue

            path = os.path.join(root, filename)
            if os.path.getsize(path) == 0:
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            modified = False

            for s in data.get("sentences", []):
                score = compute_ocr_quality(s.get("text", ""), combined_vocab)

                #add OCR quality to sentence
                s["ocr_quality"] = round(score, 2)

                modified = True

                if match_type == "exact":
                    exact_scores.append(score)
                else:
                    fuzzy_scores.append(score)

            #write back to the same JSON file
            if modified:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)


    print(f"Exact sentences: {len(exact_scores)}")
    print(f"Fuzzy sentences: {len(fuzzy_scores)}")

    # ---- plot ----
    plt.figure()
    plt.hist(exact_scores, bins=20, alpha=0.7, label="Exact sentences")
    plt.hist(fuzzy_scores, bins=20, alpha=0.7, label="Fuzzy sentences")
    plt.xlabel("OCR quality (% tokens recognised)")
    plt.ylabel("Number of sentences")
    plt.title("Sentence-level OCR quality: Exact vs Fuzzy")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
