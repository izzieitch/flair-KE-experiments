import argparse
import json
import logging
from pathlib import Path
 
from flair.data import Sentence
from flair.models import SequenceTagger
 
# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
 
 
# ── folder traversal ──────────────────────────────────────────────────────────
def iter_sentences(input_dir: Path):
    """Yield (metadata, sentence_dict) from all JSON files recursively."""
    json_files = sorted(input_dir.rglob("*.json"))
    if not json_files:
        logger.warning(f"No JSON files found under {input_dir}")
 
    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read {jf}: {e}")
            continue
 
        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            metadata = {k: v for k, v in entry.items() if k != "sentences"}
            for sent in entry.get("sentences", []):
                yield metadata, sent
 
 
# ── NER helper ────────────────────────────────────────────────────────────────
def run_flair_ner(tagger: SequenceTagger, text: str) -> list[dict]:
    """
    Run Flair NER on a single text string.
    Returns a list of entity dicts:
        {text, label, start_pos, end_pos, score}
    """
    sentence = Sentence(text)
    tagger.predict(sentence)
 
    entities = []
    for span in sentence.get_spans("ner"):
        entities.append(
            {
                "text": span.text,
                "label": span.get_label("ner").value,
                "start_pos": span.start_position,
                "end_pos": span.end_position,
                "score": round(span.get_label("ner").score, 4),
            }
        )
    return entities
 
 
# ── main ──────────────────────────────────────────────────────────────────────
def main(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
 
    logger.info("Loading flair/ner-english-ontonotes-large …")
    tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")
    logger.info("Model loaded.")
 
    all_results = []   # single flat list : all sentences across all files
 
    for metadata, sent in iter_sentences(input_dir):
        text = sent.get("text", "").strip()
        if not text:
            continue
 
        entities = run_flair_ner(tagger, text)
 
        record = {
            **metadata,                                                    # source_file, target_concept, match_type, …
            "sent_id":     sent.get("sentence_id") or sent.get("sent_id") or sent.get("id", ""),
            "ocr_quality": sent.get("ocr_quality"),                        # float e.g. 78.26, or None
            "text":        text,
            "entities":    entities,
        }
 
        all_results.append(record)
 
    if not all_results:
        logger.warning("No sentences were processed. Check your input files.")
        return
 
    # Write everything to one single file
    out_path = output_dir / "flair_ner_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
 
    total_ents = sum(len(r["entities"]) for r in all_results)
    logger.info(
        f"Done. {len(all_results)} sentences | "
        f"{total_ents} entities → {out_path}"
    )
 
 
# ── CLI ───────────────────────────────────────────────────────────────────────
DEFAULT_INPUT_DIR  = Path("opium_sentences_complete")
DEFAULT_OUTPUT_DIR = Path("results")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply flair/ner-english-ontonotes-large to a folder of JSON files."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Root folder containing input JSON files (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder where output JSON files will be saved (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)