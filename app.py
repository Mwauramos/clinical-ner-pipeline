# app.py
# Streamlit UI for Clinical NER Pipeline
# ZENIK.AI — Computational Epidemiology Portfolio

import streamlit as st
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

st.set_page_config(
    page_title="Clinical NER — East African Epidemiology",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Clinical Entity Extraction")
st.markdown(
    "Extract structured epidemiological entities from research abstracts. "
    "Built for East African health literature using a hybrid rule-based and "
    "biomedical transformer pipeline."
)
st.divider()

@st.cache_resource
def load_pipelines():
    from ner_pipeline import build_ner_pipeline
    from scibert_ner import load_biomedical_ner
    nlp, phrase_matcher = build_ner_pipeline()
    ner_pipeline = load_biomedical_ner()
    return nlp, phrase_matcher, ner_pipeline

with st.spinner("Loading NER pipelines..."):
    nlp, phrase_matcher, ner_pipeline = load_pipelines()

st.success("Pipelines ready.")

# Sample abstracts
SAMPLES = {
    "Malaria — Kenya": (
        "A cross-sectional study conducted in Kisumu County, Kenya examined "
        "malaria prevalence among children under five years. Results showed "
        "that iron deficiency anaemia was associated with a threefold increased "
        "risk of severe malaria. Plasmodium falciparum was the dominant species "
        "accounting for 87 percent of confirmed cases. Insecticide-treated "
        "bed nets were used by only 34 percent of households."
    ),
    "TB — Nairobi": (
        "A retrospective cohort study assessed tuberculosis treatment success "
        "rates in Kibera and Mathare informal settlements in Nairobi, Kenya. "
        "HIV co-infection was identified as the strongest predictor of treatment "
        "failure, with an odds ratio of 3.2. The study followed 1,200 TB patients "
        "over a 12-month period."
    ),
    "Maternal Mortality — Tanzania": (
        "A case-control study in Dodoma and Mwanza regions of Tanzania "
        "investigated determinants of maternal mortality. Lack of skilled "
        "birth attendance was the primary risk factor. Postpartum haemorrhage "
        "and eclampsia were the leading direct causes of death."
    )
}

st.markdown("**Try a sample abstract or paste your own:**")
sample_choice = st.selectbox("Sample abstracts:", ["Paste my own"] + list(SAMPLES.keys()))

if sample_choice != "Paste my own":
    default_text = SAMPLES[sample_choice]
else:
    default_text = ""

abstract_text = st.text_area(
    "Research abstract:",
    value=default_text,
    height=200,
    placeholder="Paste any epidemiological research abstract here..."
)

if st.button("Extract Entities", type="primary") and abstract_text:
    with st.spinner("Extracting entities..."):
        from ner_pipeline import extract_entities
        from scibert_ner import extract_biomedical_entities
        from hybrid_pipeline import clean_scibert_entities, combine_entities

        rule_entities = extract_entities(abstract_text, nlp, phrase_matcher)
        scibert_raw = extract_biomedical_entities(abstract_text, ner_pipeline)
        scibert_clean = clean_scibert_entities(scibert_raw)
        combined = combine_entities(rule_entities, scibert_clean)

    st.markdown("### Extracted Entities")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Diseases & Conditions**")
        if combined["diseases"]:
            for d in combined["diseases"]:
                st.markdown(f"- {d}")
        else:
            st.markdown("_None found_")

        st.markdown("**Study Design**")
        if combined["study_design"]:
            for s in combined["study_design"]:
                st.markdown(f"- {s}")
        else:
            st.markdown("_None found_")

        st.markdown("**Populations**")
        if combined["populations"]:
            for p in combined["populations"]:
                st.markdown(f"- {p}")
        else:
            st.markdown("_None found_")

        st.markdown("**Risk Factors**")
        if combined["risk_factors"]:
            for r in combined["risk_factors"]:
                st.markdown(f"- {r}")
        else:
            st.markdown("_None found_")

    with col2:
        st.markdown("**Locations**")
        if combined["locations"]:
            for l in combined["locations"]:
                st.markdown(f"- {l}")
        else:
            st.markdown("_None found_")

        st.markdown("**Interventions**")
        if combined["interventions"]:
            for i in combined["interventions"]:
                st.markdown(f"- {i}")
        else:
            st.markdown("_None found_")

        st.markdown("**Statistics**")
        if combined["statistics"]:
            for s in combined["statistics"]:
                st.markdown(f"- {s}")
        else:
            st.markdown("_None found_")

        st.markdown("**Time References**")
        if combined["time_references"]:
            for t in combined["time_references"]:
                st.markdown(f"- {t}")
        else:
            st.markdown("_None found_")

    st.divider()
    st.markdown("### Raw JSON Output")
    st.json(combined)

    json_str = json.dumps(combined, indent=2)
    st.download_button(
        label="Download JSON",
        data=json_str,
        file_name="extracted_entities.json",
        mime="application/json"
    )

st.divider()
st.caption("ZENIK.AI · Clinical NER Pipeline · Built with spaCy and Hugging Face Transformers")