import json
import spacy
import os

INPUT_JSON = "results/flair_ner_with_manual.json"
OUTPUT_JSON = "results/hybrid_spacy_on_flair.json"

nlp = spacy.load("en_core_web_lg")

MIN_ENTITY_SCORE = 0.5

# High-quality rule-based relations
RELATION_MAP = {
    ("nsubj", "dobj"): "ACTS_ON",
    ("nsubjpass", "pobj"): "AFFECTED_BY",
}

STOP_VERBS = {"be", "have", "do"}


# -------------------------
# ENTITY HANDLING
# -------------------------
def build_entity_spans(doc, entities):
    spans = []

    for ent in entities:
        if ent.get("score", 1.0) < MIN_ENTITY_SCORE:
            continue

        span = doc.char_span(ent["start_pos"], ent["end_pos"])

        if span:
            spans.append({
                "text": span.text,
                "label": ent["label"],
                "span": span,
                "kb_url": ent.get("wikipedia_url")
            })

    return spans


def find_entity(token, spans):
    for ent in spans:
        if ent["span"].start <= token.i < ent["span"].end:
            return ent
    return None


# -------------------------
# MAIN EXTRACTION
# -------------------------
def extract_triples(text, entities):
    doc = nlp(text)
    spans = build_entity_spans(doc, entities)

    triples = []

    for token in doc:

        # =========================
        # 1. STRICT SVO (mapped)
        # =========================
        if token.pos_ == "VERB":

            subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
            objects = [c for c in token.children if c.dep_ in ("dobj", "pobj")]

            for subj in subjects:
                for obj in objects:

                    subj_ent = find_entity(subj, spans)
                    obj_ent = find_entity(obj, spans)

                    if not subj_ent or not obj_ent:
                        continue

                    key = (subj.dep_, obj.dep_)

                    if key in RELATION_MAP:
                        triples.append({
                            "subject": subj_ent["text"],
                            "relation": RELATION_MAP[key],
                            "relation_type": "mapped",
                            "object": obj_ent["text"],
                            "verb": token.lemma_
                        })

        # =========================
        # 2. EXPANDED VERB RELATIONS (open)
        # =========================
        if token.pos_ == "VERB":

            verb_lemma = token.lemma_.lower()

            if verb_lemma in STOP_VERBS:
                continue

            connected_entities = []

            # children
            for child in token.children:
                ent = find_entity(child, spans)
                if ent:
                    connected_entities.append(ent)

            # head (important!)
            head_ent = find_entity(token.head, spans)
            if head_ent:
                connected_entities.append(head_ent)

            # remove duplicates
            unique_entities = list({e["text"]: e for e in connected_entities}.values())

            # create pairwise relations
            for i in range(len(unique_entities)):
                for j in range(i + 1, len(unique_entities)):

                    e1 = unique_entities[i]
                    e2 = unique_entities[j]

                    triples.append({
                        "subject": e1["text"],
                        "relation": verb_lemma.upper(),
                        "relation_type": "open",
                        "object": e2["text"],
                        "verb": verb_lemma
                    })

        # =========================
        # 3. PREPOSITIONAL RELATIONS (open)
        # =========================
        if token.dep_ == "prep":

            head = token.head
            pobj_list = [c for c in token.children if c.dep_ == "pobj"]

            for obj in pobj_list:

                subj_ent = find_entity(head, spans)
                obj_ent = find_entity(obj, spans)

                if not subj_ent or not obj_ent:
                    continue

                triples.append({
                    "subject": subj_ent["text"],
                    "relation": token.text.upper(),
                    "relation_type": "open",
                    "object": obj_ent["text"],
                    "verb": token.text
                })

        # =========================
        # 4. APPOSITIONAL RELATIONS (mapped)
        # =========================
        if token.dep_ == "appos":

            head = token.head

            subj_ent = find_entity(head, spans)
            obj_ent = find_entity(token, spans)

            if not subj_ent or not obj_ent:
                continue

            triples.append({
                "subject": subj_ent["text"],
                "relation": "IS_A",
                "relation_type": "mapped",
                "object": obj_ent["text"],
                "verb": "is"
            })

    return triples


# -------------------------
# MAIN
# -------------------------
def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for row in data:

        text = row["text"]
        entities = row.get("entities", [])

        if len(entities) < 2:
            continue

        triples = extract_triples(text, entities)

        for t in triples:
            results.append({
                "source_file": row["source_file"],
                "sent_id": row["sent_id"],
                "concept": row["target_concept"],
                "text": text,
                "match_type": row["match_type"],
                **t
            })

    os.makedirs("results", exist_ok=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(results)} triples.")
    print(f"Saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()