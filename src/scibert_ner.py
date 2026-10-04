# scibert_ner.py
# SciBERT-based NER for epidemiological entity extraction
# ZENIK.AI — Computational Epidemiology Portfolio

import sys
import os
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.abstracts import ABSTRACTS

# ── What this does ──────────────────────────────────────────────────────────
# We load a biomedical NER model from Hugging Face trained on scientific text
# It uses BERT architecture fine-tuned to recognise biomedical entities
# Unlike rule-based, it finds entities it was never explicitly told about
# ────────────────────────────────────────────────────────────────────────────

def load_biomedical_ner():
    """Load a biomedical NER model from Hugging Face."""
    print("Loading biomedical NER model...")
    print("This may take 2 to 3 minutes on first run (downloading model weights)...")

    model_name = "d4data/biomedical-ner-all"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )

    print("Biomedical NER model loaded.")
    return ner_pipeline


def extract_biomedical_entities(text: str, ner_pipeline) -> dict:
    """Extract biomedical entities using SciBERT-style model."""
    results = ner_pipeline(text)

    entities = {}
    for result in results:
        label = result["entity_group"]
        word = result["word"]
        score = round(result["score"], 3)

        if label not in entities:
            entities[label] = []

        if word not in [e["text"] for e in entities[label]]:
            entities[label].append({"text": word, "score": score})

    return entities


def display_biomedical_results(abstract: dict, entities: dict):
    """Display extracted biomedical entities."""
    print(f"\n{'='*60}")
    print(f"ID: {abstract['id']} | {abstract['title']}")
    print(f"{'='*60}")

    if not entities:
        print("No entities found.")
        return

    for label, items in entities.items():
        texts = [f"{e['text']} ({e['score']})" for e in items[:5]]
        print(f"{label:20}: {', '.join(texts)}")

    print()


if __name__ == "__main__":
    ner_pipeline = load_biomedical_ner()

    print(f"\nExtracting biomedical entities from {len(ABSTRACTS)} abstracts...\n")

    for abstract in ABSTRACTS:
        entities = extract_biomedical_entities(abstract["text"], ner_pipeline)
        display_biomedical_results(abstract, entities)

    print("Day 2 Project 2 complete — SciBERT NER pipeline working.")