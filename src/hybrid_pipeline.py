# hybrid_pipeline.py
# Combines rule-based and SciBERT NER into one structured output
# ZENIK.AI — Computational Epidemiology Portfolio

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.abstracts import ABSTRACTS
from ner_pipeline import build_ner_pipeline, extract_entities
from scibert_ner import load_biomedical_ner, extract_biomedical_entities


def clean_scibert_entities(entities: dict) -> dict:
    """Clean SciBERT output by removing subword fragments and low confidence."""
    cleaned = {}
    for label, items in entities.items():
        clean_items = []
        for item in items:
            text = item["text"]
            score = item["score"]
            # Skip subword fragments and low confidence
            if text.startswith("##"):
                continue
            if score < 0.5:
                continue
            if len(text) < 2:
                continue
            clean_items.append({"text": text, "score": score})
        if clean_items:
            cleaned[label] = clean_items
    return cleaned


def combine_entities(rule_entities: dict, scibert_entities: dict) -> dict:
    """Merge rule-based and SciBERT entities into unified categories."""
    
    # Map SciBERT categories to our unified categories
    scibert_disease_labels = {"Disease_disorder", "Sign_symptom"}
    scibert_intervention_labels = {"Therapeutic_procedure", "Medication"}
    scibert_stats_labels = {"Lab_value", "Quantitative_concept"}
    scibert_time_labels = {"Duration", "Date"}

    combined = {
        "diseases": list(rule_entities.get("DISEASE", [])),
        "locations": list(rule_entities.get("LOCATION", [])),
        "study_design": list(rule_entities.get("STUDY_DESIGN", [])),
        "populations": list(rule_entities.get("POPULATION", [])),
        "risk_factors": list(rule_entities.get("RISK_FACTOR", [])),
        "interventions": [],
        "statistics": [],
        "time_references": []
    }

    # Add SciBERT diseases not already found by rules
    for label in scibert_disease_labels:
        for item in scibert_entities.get(label, []):
            text = item["text"]
            if text.lower() not in [d.lower() for d in combined["diseases"]]:
                combined["diseases"].append(text)

    # Add interventions from SciBERT
    for label in scibert_intervention_labels:
        for item in scibert_entities.get(label, []):
            combined["interventions"].append(item["text"])

    # Add statistics from SciBERT
    for label in scibert_stats_labels:
        for item in scibert_entities.get(label, []):
            combined["statistics"].append(item["text"])

    # Add time references from SciBERT
    for label in scibert_time_labels:
        for item in scibert_entities.get(label, []):
            combined["time_references"].append(item["text"])

    # Remove duplicates
    for key in combined:
        seen = set()
        combined[key] = [
            x for x in combined[key]
            if x.lower() not in seen and not seen.add(x.lower())
        ]

    return combined


def process_abstract(abstract: dict, nlp, phrase_matcher, ner_pipeline) -> dict:
    """Run full hybrid NER on a single abstract."""
    text = abstract["text"]

    rule_entities = extract_entities(text, nlp, phrase_matcher)
    scibert_raw = extract_biomedical_entities(text, ner_pipeline)
    scibert_clean = clean_scibert_entities(scibert_raw)
    combined = combine_entities(rule_entities, scibert_clean)

    return {
        "id": abstract["id"],
        "title": abstract["title"],
        "source": abstract["source"],
        "rule_based": rule_entities,
        "scibert": {k: [i["text"] for i in v] for k, v in scibert_clean.items()},
        "combined": combined
    }


def display_combined(result: dict):
    """Display the combined structured output."""
    print(f"\n{'='*60}")
    print(f"ID: {result['id']} | {result['title']}")
    print(f"{'='*60}")
    combined = result["combined"]
    for category, items in combined.items():
        if items:
            print(f"{category:20}: {', '.join(items)}")
    print()


if __name__ == "__main__":
    print("Loading pipelines...")
    nlp, phrase_matcher = build_ner_pipeline()
    ner_pipeline = load_biomedical_ner()

    print(f"\nRunning hybrid NER on {len(ABSTRACTS)} abstracts...\n")

    all_results = []
    for abstract in ABSTRACTS:
        result = process_abstract(abstract, nlp, phrase_matcher, ner_pipeline)
        display_combined(result)
        all_results.append(result)

    # Save to JSON
    output_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'extracted_entities.json'
    )
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to data/extracted_entities.json")
    print("Day 3 Project 2 complete — hybrid NER pipeline working.")