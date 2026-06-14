import argparse
import json
import logging
from pathlib import Path

from flair.data import Sentence
from flair.nn import Classifier

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── default paths ─────────────────────────────────────────────────────────────
DEFAULT_INPUT_FILE = Path("results/flair_ner_with_manual.json")
DEFAULT_OUTPUT_DIR = Path("results")


# ── load NER results (single file) ────────────────────────────────────────────
def load_ner_records(input_file: Path):
    """
    Load records from a single JSON file.

    Returns:
        source_stem: filename without extension
        records: list of dicts
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    try:
        with open(input_file, encoding="utf-8") as f:
            records = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Could not read {input_file}: {e}")

    if not isinstance(records, list):
        records = [records]

    return input_file.stem, records


# ── entity linking helper ─────────────────────────────────────────────────────
def link_entities(
    linker: Classifier,
    text: str,
    ner_entities: list[dict],
) -> list[dict]:

    if not ner_entities:
        return ner_entities

    sentence = Sentence(text)
    linker.predict(sentence)

    linking_map: dict[str, dict] = {}

    for label in sentence.get_labels():
        span_text = label.data_point.text
        wiki_value = label.value

        if wiki_value and wiki_value != "O":
            linking_map[span_text] = {
                "wikipedia_title": wiki_value.replace("_", " "),
                "wikipedia_url": f"https://en.wikipedia.org/wiki/{wiki_value}",
                "linking_score": round(label.score, 4),
            }
        else:
            linking_map.setdefault(span_text, {
                "wikipedia_title": None,
                "wikipedia_url": None,
                "linking_score": None,
            })

    null_link = {
        "wikipedia_title": None,
        "wikipedia_url": None,
        "linking_score": None,
    }

    enriched = []
    for ent in ner_entities:
        link_info = linking_map.get(ent["text"], null_link)
        enriched.append({**ent, **link_info})

    return enriched


# ── main ──────────────────────────────────────────────────────────────────────
def main(input_file: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Flair entity linker (Classifier 'linker') …")
    linker = Classifier.load("linker")
    logger.info("Linker loaded.")

    source_stem, records = load_ner_records(input_file)

    results = []

    for record in records:
        text = record.get("text", "").strip()
        ner_entities = record.get("entities", [])

        if not text:
            continue

        try:
            enriched_entities = link_entities(linker, text, ner_entities)
        except Exception as e:
            logger.warning(f"Linking failed for sent '{text[:60]}…': {e}")
            enriched_entities = ner_entities

        linked_record = {
            **{k: v for k, v in record.items() if k != "entities"},
            "entities": enriched_entities,
        }

        results.append(linked_record)

    if not results:
        logger.warning("No records were processed.")
        return

    out_path = output_dir / f"{source_stem}_linked.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(results)} sentences → {out_path}")

    total_sents = len(results)
    total_ents = sum(len(r["entities"]) for r in results)
    linked_ents = sum(
        1
        for r in results
        for e in r["entities"]
        if e.get("wikipedia_url")
    )

    logger.info(
        f"Done. {total_sents} sentences | {total_ents} entities | "
        f"{linked_ents} successfully linked to Wikipedia."
    )


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Link NER entities to Wikipedia using Flair's Classifier linker."
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"Input JSON file (default: {DEFAULT_INPUT_FILE}).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder to write linked output JSON (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    main(args.input_file, args.output_dir)