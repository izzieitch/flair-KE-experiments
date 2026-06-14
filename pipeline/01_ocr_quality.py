import os
from nltk.tokenize import word_tokenize
from nltk.corpus import words


TEXT_FOLDER = "travelogues_txt" #be sure to populate this folder with .txt files of the travelogues found in corpus-info.xlsx before running this script.
GUTENBERG_EXTRA_VOCAB = "vocab/gutenberg_not_in_nltk_words.txt"
EXPANDED_NLTK = "vocab/nltk_words_expanded.txt"



with open(EXPANDED_NLTK, "r", encoding="utf-8") as f:
    expanded_nltk = set(line.strip() for line in f if line.strip())

with open(GUTENBERG_EXTRA_VOCAB, "r", encoding="utf-8") as f:
    gutenberg_extra_vocab = set(line.strip() for line in f if line.strip())


combined_vocab = expanded_nltk | gutenberg_extra_vocab


def compute_ocr_quality(text, vocab):
    tokens = word_tokenize(text)
    if not tokens:
        return 0.0

    recognized_count = sum(1 for t in tokens if t.lower() in vocab)
    return (recognized_count / len(tokens)) * 100

def get_token_type_counts(text):
    tokens = word_tokenize(text)
    word_tokens = [t for t in tokens if t.isalpha()]
    token_count = len(word_tokens)
    type_count = len(set(t.lower() for t in word_tokens))
    return token_count, type_count

def main():
    print("OCR Quality Report (% tokens recognised)\n")
    print("Note:")
    print(" - Baseline: nltk.words expanded")
    print(" - Extended: nltk.words expanded + Gutenberg additions\n")

    total_tokens = 0
    total_types_combined = set()

    for filename in sorted(os.listdir(TEXT_FOLDER)):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(TEXT_FOLDER, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
          
        baseline_score = compute_ocr_quality(text, expanded_nltk)
        extended_score = compute_ocr_quality(text, combined_vocab)
        improvement = extended_score - baseline_score

        token_count, type_count = get_token_type_counts(text)
        total_tokens += token_count
        total_types_combined.update(
            t.lower() for t in word_tokenize(text) if t.isalpha()
        )

        print(
            f"{filename}: "
            f"{baseline_score:.2f}% → {extended_score:.2f}% "
            f"(+{improvement:.2f}%) | "
            f"Tokens: {token_count:,} | Types: {type_count:,}"
        )

    print(f"\nCorpus totals — Tokens: {total_tokens:,} | Types: {len(total_types_combined):,}")

if __name__ == "__main__":
    main()