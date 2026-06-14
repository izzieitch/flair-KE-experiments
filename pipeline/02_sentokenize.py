import os
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

#------------top level ----------
TEXT_FOLDER = "travelogues_txt"
OUTPUT_FOLDER = "tokenizedSentences"
LANGUAGE = "english"

#----------------------------------

def setup_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

def tokenize_file(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    sentences = sent_tokenize(text, language=LANGUAGE)

    tokenized = []
    for sent_id, sentence in enumerate(sentences):
        tokens = word_tokenize(sentence, language=LANGUAGE)
        tokenized.append({
            "sentence_id": sent_id,
            "text": sentence,
            "tokens": tokens
        })

    return tokenized

def main():
    setup_nltk()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for filename in os.listdir(TEXT_FOLDER):
        if not filename.endswith(".txt"):
            continue

        input_path = os.path.join(TEXT_FOLDER, filename)
        output_name = filename.replace(".txt", ".json")
        output_path = os.path.join(OUTPUT_FOLDER, output_name)

        tokenized_data = tokenize_file(input_path)

        with open(output_path, "w", encoding="utf-8") as out:
            json.dump(tokenized_data, out, indent=2, ensure_ascii=False)

        print(f"✓ {filename} → {output_path}")

if __name__ == "__main__":
    main()

