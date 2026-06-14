from nltk.corpus import gutenberg, words
from nltk.tokenize import word_tokenize
from pathlib import Path
import spacy
import pyinflect

# Load spaCy with pyinflect
nlp = spacy.load("en_core_web_sm")

GUTENBERG_FILES = [
    "austen-emma.txt",
    "austen-persuasion.txt",
    "austen-sense.txt",
    "carroll-alice.txt",
    "melville-moby_dick.txt",
    "whitman-leaves.txt"
]

def extract_types(file_ids):
    vocab = set()

    for file_id in file_ids:
        text = gutenberg.raw(file_id)
        tokens = word_tokenize(text)

        for token in tokens:
            # keep alphabetic tokens only
            if token.isalpha():
                vocab.add(token.lower())

    return vocab

def load_nltk_words():
    return set(w.lower() for w in words.words())

def expand_vocab_morphology(vocab):
    """Expand vocabulary to include plurals, tenses, and participles using pyinflect."""
    expanded = set()
    for w in vocab:
        expanded.add(w)  # original nltk.words

        # Use spaCy + pyinflect to get morphological forms
        doc = nlp(w)
        token = doc[0]

        for tag in ["VB", "VBD", "VBG", "VBN", "VBZ"]:
            form = token._.inflect(tag)
            if form:
                expanded.add(form.lower())

        # Handle nouns (plural)
        plural = token._.inflect("NNS")
        if plural:
            expanded.add(plural.lower())
    return expanded


def save_txt(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for word in sorted(data):
            f.write(word + "\n")

def main():
    gutenberg_vocab = extract_types(GUTENBERG_FILES)
    nltk_vocab = load_nltk_words()

    # Expand NLTK words with WordNet variants
    nltk_vocab_expanded = expand_vocab_morphology(nltk_vocab)

    missing_from_nltk = gutenberg_vocab - nltk_vocab_expanded

    print(f"Gutenberg unique types: {len(gutenberg_vocab)}")
    print(f"Missing from nltk.words + morphological expansion: {len(missing_from_nltk)}")

    save_txt(gutenberg_vocab, "vocab/gutenberg_types.txt")
    save_txt(missing_from_nltk, "vocab/gutenberg_not_in_nltk_words.txt")
    save_txt(nltk_vocab_expanded, "vocab/nltk_words_expanded.txt")

    print("Saved:")
    print(" - vocab/gutenberg_types.txt")
    print(" - vocab/gutenberg_not_in_nltk_words.txt")
    print(" - vocab/nltk_words_expanded.txt")
if __name__ == "__main__":
    main()