import spacy
from spacy.matcher import PhraseMatcher
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.abstracts import ABSTRACTS


def build_ner_pipeline():
    nlp = spacy.load("en_core_web_sm")
    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    diseases = [
        "malaria", "tuberculosis", "tb", "hiv", "aids", "maternal mortality",
        "anaemia", "iron deficiency anaemia", "severe malaria",
        "multidrug-resistant tb", "postpartum haemorrhage", "eclampsia",
        "plasmodium falciparum"
    ]
    study_designs = [
        "cross-sectional study", "retrospective cohort study",
        "prospective cohort study", "case-control study",
        "randomised controlled trial", "systematic review", "meta-analysis"
    ]
    populations = [
        "children under five", "adolescents", "adult patients",
        "female adolescents", "male participants", "women",
        "hiv-negative adolescents"
    ]
    risk_factors = [
        "iron deficiency", "hiv co-infection", "lack of skilled birth attendance",
        "early sexual debut", "low condom use",
        "distance greater than 10 kilometres"
    ]

    phrase_matcher.add("DISEASE", [nlp.make_doc(t) for t in diseases])
    phrase_matcher.add("STUDY_DESIGN", [nlp.make_doc(t) for t in study_designs])
    phrase_matcher.add("POPULATION", [nlp.make_doc(t) for t in populations])
    phrase_matcher.add("RISK_FACTOR", [nlp.make_doc(t) for t in risk_factors])

    print("NER pipeline ready.")
    return nlp, phrase_matcher


def extract_entities(text, nlp, phrase_matcher):
    doc = nlp(text)
    matches = phrase_matcher(doc)

    entities = {
        "DISEASE": [],
        "STUDY_DESIGN": [],
        "POPULATION": [],
        "RISK_FACTOR": [],
        "LOCATION": []
    }

    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        span = doc[start:end].text
        if span not in entities[label]:
            entities[label].append(span)

    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC") and ent.text not in entities["LOCATION"]:
            entities["LOCATION"].append(ent.text)

    return entities


def display_results(abstract, entities):
    print(f"\n{'='*60}")
    print(f"ID: {abstract['id']} | {abstract['title']}")
    print(f"Source: {abstract['source']}")
    print(f"{'='*60}")
    for label, items in entities.items():
        if items:
            print(f"{label:15}: {', '.join(items)}")
    print()


if __name__ == "__main__":
    nlp, phrase_matcher = build_ner_pipeline()
    print(f"\nProcessing {len(ABSTRACTS)} abstracts...\n")
    for abstract in ABSTRACTS:
        entities = extract_entities(abstract["text"], nlp, phrase_matcher)
        display_results(abstract, entities)
    print("Project 2 Day 1 complete.")