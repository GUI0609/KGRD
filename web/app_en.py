import streamlit as st
import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import requests
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

sys.path.append("/share/gguilin/rd-project/pheval")

from pheval_utils import *
sys.path.append("/share/gguilin/rd-project/Multi-agent-conversation-for-disease-diagnosis/utils")
from utils import *
from data import query_one_hop_gene_disease, merge_and_sort

# =======================
# Global Configuration
# =======================

BASE_DIR = "/share/gguilin/rd-project/pheval/public_run/output_dir/list_output"
BASE_URL = "http://127.0.0.1:10203"
CHAT_API_URL = "http://127.0.0.1:8008/api/run"
TXT2HPO_URL = "http://127.0.0.1:10203/txt2hpo"

# =======================
# Core Utility Functions
# =======================

def call_txt2hpo(text, method="scispacy", timeout=60):
    if not text or not str(text).strip():
        return {"hpo_list": [], "hpo_name_list": [], "translated_text": ""}

    payload = {"text": text, "method": method}
    try:
        resp = requests.post(TXT2HPO_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Text-to-HPO service error: {e}")
        return {"hpo_list": [], "hpo_name_list": [], "translated_text": ""}

def get_top_genes(sample_id, tool_prefix, n=None):
    file_name = f"{tool_prefix}_list_{sample_id}.txt"
    file_path = os.path.join(BASE_DIR, file_name)
    if not os.path.exists(file_path):
        return []

    genes = []
    with open(file_path, "r") as f:
        for line in f:
            g = line.strip()
            if g:
                genes.append(g)
            if n and len(genes) >= n:
                break
    return genes

def _post_json(path, payload):
    url = BASE_URL + path
    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.warning(f"API request failed ({path}): {e}")
        return []

def run_phenotype_to_disease(hpo_name_list):
    return _post_json("/query_phenotype_to_disease", {"phenotype_ids": hpo_name_list}) if hpo_name_list else []

def run_phenobrain(hpo_list):
    return _post_json("/PhenoBrain", {"hpo_list": hpo_list}) if hpo_list else []

def run_g2d(genes):
    return _post_json("/g2d", {"genes": genes}) if genes else []

def run_sapbert(hpo_name_list):
    return _post_json("/sapbert_d_patient", {"hpo_name_list": hpo_name_list}) if hpo_name_list else []

def build_payload(sample_id, presentation, hpo_list, hpo_name_list, genes, d_pheno, d_patient, d_gene):
    return {
        "sample_cases": [
            {
                "crl": sample_id,
                "initial_presentation": presentation,
                "case_type": "custom",
                "hpo_ids": hpo_list,
                "hpo_names": hpo_name_list,
                "genes": genes,
                "priors": {
                    "from_phenotype": d_pheno,
                    "from_patient_like_me": d_patient,
                    "from_gene": d_gene
                }
            }
        ]
    }

# =======================
# Streamlit UI
# =======================

st.set_page_config(
    page_title="KGRD: An Automated Rare Disease Diagnostic System",
    layout="wide"
)

st.title("KGRD: Knowledge Graph–Driven Rare Disease Diagnosis System")

# =======================
# Sidebar: Configuration
# =======================

with st.sidebar:
    st.header("Configuration Parameters")

    sample_id_input = st.text_input(
        "Sample Identifier",
        value="BC243043OO"
    )

    default_vcf = f"rd-project/SHEPHERD/xmu_vcf/{sample_id_input}.vcf"
    vcf_path_input = st.text_input(
        "Path to VCF File",
        value=default_vcf
    )

    st.subheader("Clinical Phenotype Description")

    default_text = (
        "Male patient with the following clinical features: "
        "1. Aortic valve stenosis with regurgitation (status post TAVR); "
        "2. Coronary atherosclerotic heart disease (status post PCI of the right coronary artery); "
        "3. Hypertrophic obstructive cardiomyopathy; "
        "4. New York Heart Association functional class II; "
        "5. Subclavian artery stenosis; "
        "6. Pulmonary emphysema."
    )

    phenotype_text_input = st.text_area(
        "Free-text Clinical Phenotype",
        value=default_text,
        height=150
    )

# =======================
# Session State
# =======================

if "step1_result" not in st.session_state:
    st.session_state.step1_result = None

# =======================
# Stage 1: Pipeline Execution
# =======================

if st.sidebar.button("Run Diagnostic Pipeline", type="primary"):
    with st.spinner(
        "Executing the diagnostic pipeline. This process may take several minutes..."
    ):
        try:
            st.info(
                f"Running rare disease analysis pipeline for sample {sample_id_input}..."
            )

            script_path = "/share/gguilin/rd-project/pheval/public_run/run_rd_pipelines.sh"
            # subprocess.run(["bash", script_path, sample_id_input, vcf_path_input], check=True)

            st.info(
                "Extracting Human Phenotype Ontology (HPO) terms from clinical narrative..."
            )
            txt_res = call_txt2hpo(phenotype_text_input)

            st.info(
                "Loading gene prioritization results from multiple computational tools..."
            )
            aim_genes = get_top_genes(sample_id_input, "aim", 20)
            exo_genes = get_top_genes(sample_id_input, "exomiser", 20)
            lir_genes = get_top_genes(sample_id_input, "lirical", 20)

            raw_merged = list(
                dict.fromkeys(
                    lir_genes
                    + query_phenotype_to_gene(txt_res.get("hpo_name_list", []))
                    + exo_genes
                    + aim_genes
                )
            )
            default_genes = raw_merged[:10]

            st.session_state.step1_result = {
                "hpo_list": txt_res.get("hpo_list", []),
                "hpo_name_list": txt_res.get("hpo_name_list", []),
                "translated_text": txt_res.get(
                    "translated_text", phenotype_text_input
                ),
                "genes": default_genes,
                "raw_tools": {
                    "AIM": aim_genes,
                    "Exomiser": exo_genes,
                    "LIRICAL": lir_genes,
                },
            }

            st.success("Pipeline execution completed successfully.")

        except Exception as e:
            st.error(f"Pipeline execution failed: {str(e)}")

# =======================
# Stage 2: Result Review
# =======================

if st.session_state.step1_result:
    res = st.session_state.step1_result

    st.divider()
    st.header("Result Review and Expert Curation")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Extracted Phenotypic Features (HPO Terms)")

        edited_hpo_names = st.data_editor(
            pd.DataFrame({"HPO Term Names": res["hpo_name_list"]}),
            num_rows="dynamic",
        )
        final_hpo_names = edited_hpo_names["HPO Term Names"].tolist()

        st.text_area(
            "HPO Identifiers (Raw Output)",
            value="; ".join(res["hpo_list"]),
            disabled=True,
        )

    with col2:
        st.subheader(
            "Candidate Gene Evidence from Multiple Prioritization Tools"
        )

        raw_tools = res.get(
            "raw_tools", {"AIM": [], "Exomiser": [], "LIRICAL": []}
        )

        t1, t2, t3 = st.tabs(
            ["AIM Output", "Exomiser Output", "LIRICAL Output"]
        )

        with t1:
            if raw_tools["AIM"]:
                st.code("\n".join(raw_tools["AIM"][:20]), language="text")
                st.caption(
                    f"Total genes identified: {len(raw_tools['AIM'])}"
                )
            else:
                st.warning("No results found or output file is missing.")

        with t2:
            if raw_tools["Exomiser"]:
                st.code("\n".join(raw_tools["Exomiser"][:20]), language="text")
                st.caption(
                    f"Total genes identified: {len(raw_tools['Exomiser'])}"
                )
            else:
                st.warning("No results found or output file is missing.")

        with t3:
            if raw_tools["LIRICAL"]:
                st.code("\n".join(raw_tools["LIRICAL"][:20]), language="text")
                st.caption(
                    f"Total genes identified: {len(raw_tools['LIRICAL'])}"
                )
            else:
                st.warning("No results found or output file is missing.")

        st.divider()

        st.markdown(
            "**Final Candidate Gene List (Expert-Validated)**"
        )
        st.info(
            "Genes can be manually edited. Please separate gene symbols using semicolons (;)."
        )

        default_gene_str = ";".join(res["genes"])
        genes_str = st.text_area(
            "Final Gene List for Inference",
            value=default_gene_str,
            height=68,
        )
        final_genes = [
            g.strip() for g in genes_str.split(";") if g.strip()
        ]

        st.subheader("Standardized English Clinical Presentation")
        final_presentation = st.text_area(
            "Clinical Summary (English)",
            value=res["translated_text"],
            height=150,
        )

    # =======================
    # Stage 3: Inference
    # =======================

    st.divider()
    st.header("Multimodal Disease Inference and Payload Review")

    if "generated_payload" not in st.session_state:
        st.session_state.generated_payload = None
    if "intermediate_results" not in st.session_state:
        st.session_state.intermediate_results = {}

    if st.button(
        "Generate Disease Inference Results and Preview Payload"
    ):
        st.session_state.generated_payload = None
        st.session_state.intermediate_results = {}

        with st.status(
            "Performing multimodal disease inference...",
            expanded=True,
        ) as status:

            st.write(
                "Inferring disease candidates based on phenotypic features..."
            )
            d_p2d = run_phenotype_to_disease(final_hpo_names)
            d_pb = run_phenobrain(res["hpo_list"])

            try:
                DISEASE_FROM_PHENOTYPE = merge_and_sort(
                    d_pb, d_p2d
                )
            except Exception as e:
                st.error(
                    f"Phenotype-based inference failed: {e}"
                )
                DISEASE_FROM_PHENOTYPE = []

            st.write(
                "Inferring disease candidates based on gene-level evidence..."
            )
            d_g2d_raw = run_g2d(final_genes)

            d_g2d_processed = []
            if isinstance(d_g2d_raw, dict) and "results" in d_g2d_raw:
                d_g2d_processed = [
                    item.get("disease_name", "Unknown")
                    for gene in final_genes
                    for item in d_g2d_raw["results"].get(gene, [])
                ]

            one_hop_res = []
            for g in final_genes:
                try:
                    one_hop_res.append(
                        query_one_hop_gene_disease(g)
                    )
                except Exception:
                    pass

            try:
                DISEASE_FROM_GENE = merge_and_sort(
                    *one_hop_res
                )[:100]
            except Exception:
                DISEASE_FROM_GENE = []

            if d_g2d_processed:
                DISEASE_FROM_GENE += d_g2d_processed[:100]

            st.write(
                "Retrieving disease associations from phenotypically similar patients..."
            )
            d_sapbert = run_sapbert(final_hpo_names)
            DISEASE_FROM_PATIENT_LIKE_ME = (
                d_sapbert[1]
                if isinstance(d_sapbert, list)
                and len(d_sapbert) > 1
                else []
            )
            DISEASE_FROM_PATIENT_LIKE_ME = list(
                dict.fromkeys(DISEASE_FROM_PATIENT_LIKE_ME)
            )

            payload = build_payload(
                sample_id_input,
                final_presentation,
                res["hpo_list"],
                final_hpo_names,
                final_genes,
                DISEASE_FROM_PHENOTYPE,
                DISEASE_FROM_PATIENT_LIKE_ME,
                DISEASE_FROM_GENE,
            )

            st.session_state.generated_payload = payload
            st.session_state.intermediate_results = {
                "pheno": DISEASE_FROM_PHENOTYPE,
                "gene": DISEASE_FROM_GENE,
                "patient": DISEASE_FROM_PATIENT_LIKE_ME,
            }

            status.update(
                label="Disease inference completed successfully.",
                state="complete",
                expanded=False,
            )

    if st.session_state.generated_payload:
        ir = st.session_state.intermediate_results

        st.subheader(
            "Summary of Disease Evidence by Information Source (Top 10)"
        )
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**From Phenotype**")
            st.json(ir["pheno"][:10] if ir["pheno"] else [])

        with c2:
            st.markdown("**From Gene**")
            st.json(ir["gene"][:10] if ir["gene"] else [])

        with c3:
            st.markdown("**From Patient**")
            st.json(ir["patient"][:10] if ir["patient"] else [])

        with st.expander("View Complete JSON Payload"):
            st.json(st.session_state.generated_payload)

        st.divider()
        col_submit, _ = st.columns([1, 4])

        with col_submit:
            if st.button(
                "Confirm and Submit for Final Diagnostic Execution",
                type="primary",
            ):
                with st.spinner(
                    "Submitting payload to diagnostic reasoning engine..."
                ):
                    try:
                        resp = requests.post(
                            CHAT_API_URL,
                            json=st.session_state.generated_payload,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            run_id = data.get("run_id")

                            st.success(
                                f"Submission successful. Execution Run ID: {run_id}"
                            )

                            st.info(
                                f"Log stream: http://127.0.0.1:8008/stream/{run_id}"
                            )
                            st.info(
                                f"Result endpoint: http://127.0.0.1:8008/api/result/{run_id}"
                            )

                            st.markdown(
                                f"""
                            - [View Execution Log Stream](http://127.0.0.1:8008/stream/{run_id})
                            - [View Final Diagnostic Results](http://127.0.0.1:8008/api/result/{run_id})
                            """
                            )
                        else:
                            st.error(
                                f"Submission failed with status code {resp.status_code}"
                            )
                            st.code(resp.text)
                    except Exception as e:
                        st.error(
                            f"Submission error: {e}"
                        )
