import json
with open('PATH/TO/config.json', 'r') as f:
    config = json.load(f)
import os
import re
import joblib
import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# Importing internal utility functions
from utils import (
    find_best_match_entity, hpname_to_hpo, query_min_subgraph_for_verifier,
    pg2d_shortest_paths, justchat, parse_json, query_ncbi_eutils,
    OrphadataAPI, get_hpo_detail, query_in_KB
)


# =========================
# 1. Data Structure Definitions
# =========================

@dataclass
class Verdict:
    """Represents the raw result from a single evidence channel."""
    source: str                          # Channel name: "db" | "gene" | "kg" | "phenotype" | "literature"
    verdict: Optional[bool]               # True (Support), False (Oppose), None (Uncertain)
    confidence: float                     # 0.0 ~ 1.0 self-assessment
    notes: str = ""                       # Short academic rationale
    extras: Dict[str, Any] = field(default_factory=dict)

    def yes(self) -> bool: return bool(self.verdict is True)
    def no(self) -> bool: return bool(self.verdict is False)

@dataclass
class CaseInput:
    """Input structure for a diagnostic case."""
    case_initial_presentation: str
    hpo_name_list: List[str]
    gene_list: List[str]
    disease: str
    hpo_list: Optional[List[str]] = None
    debug: bool = False

@dataclass
class ChannelEvidence:
    """Structured explanation object for frontend display/audit."""
    source: str
    verdict: Optional[bool]
    confidence: float
    score: Optional[float] = None        # Feature value (1.0/0.5/0.0) for the second-layer model
    rationale: str = ""
    matched_signals: Dict[str, Any] = field(default_factory=dict)
    citations: List[Dict[str, str]] = field(default_factory=list)
    prompt_snapshot: Dict[str, Any] = field(default_factory=dict)
    raw_model_io: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FinalExplanation:
    """Comprehensive explanation package for the final decision."""
    case_fingerprint: Dict[str, Any]
    channel_evidences: List[ChannelEvidence]
    weighted_note: str
    features_used: Dict[str, float]
    decision: bool


# =========================
# 2. Normalization Logic
# =========================

def normalize_case(case: CaseInput) -> Tuple[CaseInput, List[str]]:
    """Standardizes HPO names, Disease names, and extracts HPO IDs."""
    log = []
    std_hpo_names = []
    for i in case.hpo_name_list:
        try:
            std_hpo_names.append(find_best_match_entity(i))
        except Exception as e:
            std_hpo_names.append(i)
            log.append(f"[normalize_hpo] {i} -> {e!r}")

    try:
        disease_std = find_best_match_entity(case.disease)
    except Exception as e:
        disease_std = case.disease
        log.append(f"[normalize_disease] {case.disease} -> {e!r}")

    hpo_ids = case.hpo_list
    if not hpo_ids:
        hpo_ids = []
        for n in std_hpo_names:
            try:
                hid = hpname_to_hpo(n)
                if hid: hpo_ids.append(hid)
            except Exception as e:
                log.append(f"[hpname_to_hpo] {n} -> {e!r}")
        hpo_ids = list(dict.fromkeys(hpo_ids))

    normalized = CaseInput(
        case_initial_presentation=case.case_initial_presentation,
        hpo_name_list=std_hpo_names,
        gene_list=case.gene_list,
        disease=disease_std,
        hpo_list=hpo_ids,
        debug=case.debug,
    )
    return normalized, log

# =========================
# 3. Evidence Channel Implementations
# =========================

def run_kg_channel(case: CaseInput) -> Verdict:
    """Knowledge Graph Channel: Checks for paths between HPO, Gene, and Disease."""
    try:
        tokens = case.hpo_name_list + case.gene_list + [case.disease]
        tokens = [find_best_match_entity(i) for i in tokens]
        subgraph = query_min_subgraph_for_verifier(tokens)
        paths = pg2d_shortest_paths(subgraph, tokens)
        
        if not paths:
            ce = ChannelEvidence(source="kg", verdict=None, confidence=0.3, rationale="No causal path found in KG.")
            return Verdict(source="kg", verdict=None, confidence=0.3, notes=ce.rationale, extras={"explain": asdict(ce)})

        prompt = f"""Assess whether the diagnosis is supported by causal paths in the knowledge graph. 
        Return STRICT JSON: {{"verdict": bool, "confidence": float, "rationale": str, "evidence": {{"key_paths": []}} }}
        Tokens: {tokens}
        Top Paths: {json.dumps(paths[:5], ensure_ascii=False)}"""

        raw_out = justchat(prompt, provider=config['LLM_PROVIDER'])
        data = parse_json(raw_out) or {}
        
        v = str(data.get("verdict", "")).lower() in {"true", "yes"}
        conf = float(data.get("confidence", 0) or 0.2)

        ce = ChannelEvidence(
            source="kg", verdict=v if data else None, confidence=conf,
            rationale=data.get("rationale", "KG analysis finished."),
            matched_signals={"key_paths": data.get("evidence", {}).get("key_paths", [])},
            prompt_snapshot={"tokens": tokens, "paths_preview": paths[:5]},
            raw_model_io=data
        )
        return Verdict(source="kg", verdict=ce.verdict, confidence=ce.confidence, notes=ce.rationale, extras={"explain": asdict(ce)})
    except Exception as e:
        return Verdict(source="kg", verdict=None, confidence=0.2, notes=f"KG Error: {e!r}")

def run_db_channel(case: CaseInput) -> Verdict:# disease correlation
    """Database Channel: Queries MedGen and Orphanet records."""
    orphanet = OrphadataAPI()
    orphacode = ""
    orphanet_result = {}
    medgen_data = {}

    try:
        medgen_data = query_ncbi_eutils(term=case.disease, db='medgen') or {}
        oc_data = orphanet.get_orphacode_by_name(case.disease) or {}
        orphacode = oc_data.get('data', {}).get('results', {}).get('ORPHAcode', '')
        if orphacode:
            orphanet_result = orphanet.get_phenotype_by_orphacode(orphacode) or {}
    except Exception: pass

    db_prompt = f"""Evaluate diagnosis support from MedGen/Orphanet records. 
    Return JSON with 'verdict'(bool), 'confidence'(float), 'rationale'.
    Disease: {case.disease}
    MedGen: {json.dumps(medgen_data)}
    Orphanet (ORPHA:{orphacode}): {json.dumps(orphanet_result)}"""

    data = parse_json(justchat(db_prompt, provider=config['LLM_PROVIDER'])) or {}
    v = str(data.get("verdict", "")).lower() in {"true", "yes"}
    
    ce = ChannelEvidence(
        source="db", verdict=v if data else None, confidence=data.get("confidence", 0.4),
        rationale=data.get("rationale", "DB synthesis complete."),
        matched_signals={"orphacode": orphacode},
        raw_model_io=data
    )
    return Verdict(source="db", verdict=ce.verdict, confidence=ce.confidence, notes=ce.rationale, 
                   extras={"orphacode": orphacode, "orphanet_result": orphanet_result, "explain": asdict(ce)})

def run_gene_channel(case: CaseInput) -> Verdict:
    """Gene-Disease Channel: Validates pathogenicity using NCBI Gene data."""
    if not case.gene_list:
        ce = ChannelEvidence(source="gene", verdict=None, confidence=0.0, rationale="No gene provided.")
        return Verdict(source="gene", verdict=None, confidence=0.0, notes="N/A", extras={"explain": asdict(ce)})

    gene = case.gene_list[0]
    ncbi_gene = {}
    try:
        ncbi_gene = query_ncbi_eutils(term=str(gene), db='gene') or {}
    except Exception: pass

    gene_prompt = f"""Assess gene-disease pathogenicity. Return JSON.
    Gene: {gene}, Disease: {case.disease}
    NCBI Record: {json.dumps(ncbi_gene)}"""

    data = parse_json(justchat(gene_prompt, provider=config['LLM_PROVIDER'])) or {}
    v = str(data.get("verdict", "")).lower() in {"true", "yes"}

    ce = ChannelEvidence(
        source="gene", verdict=v if data else None, confidence=data.get("confidence", 0.3),
        rationale=data.get("rationale", "Gene validation finished."),
        raw_model_io=data
    )
    return Verdict(source="gene", verdict=ce.verdict, confidence=ce.confidence, notes=ce.rationale, extras={"explain": asdict(ce)})

def run_phenotype_channel(case: CaseInput, orphanet_result: dict = None) -> Verdict:
    """Phenotype Channel: Calculates Jaccard similarity between patient and disease HPOs."""
    disease_hpos = []
    # Extracting HPO IDs from Orphanet structure
    if orphanet_result:
        results = orphanet_result.get("data", {}).get("results", [])
        if isinstance(results, list):
            for it in results:
                hid = it.get("HPOId") or (it.get("HPO") or {}).get("HPOId")
                if hid: disease_hpos.append(hid)
    if not disease_hpos:
        try:
            orph = OrphadataAPI()
            hpo_list = [i['HPOId'] for i in [item['HPO'] for item in orph.get_phenotype_by_orphacode(orph.get_orphacode_by_name(case.disease)['data']['results']['ORPHAcode'])['data']['results']['Disorder']['HPODisorderAssociation']]]
            disease_hpos = hpo_list
        except:
            pass

    patient_hpos = case.hpo_list or []
    intersection = set(patient_hpos) & set(disease_hpos)
    union = set(patient_hpos) | set(disease_hpos)
    sim = len(intersection) / len(union) if union else 0.0

    phen_prompt = f"""Compare patient HPOs vs Disease HPOs. Return JSON.
    Patient: {patient_hpos}, Disease: {disease_hpos}, Jaccard: {sim:.2f}"""

    data = parse_json(justchat(phen_prompt, provider=config['LLM_PROVIDER'])) or {}
    v = str(data.get("verdict", "")).lower() in {"true", "yes"}

    ce = ChannelEvidence(
        source="phenotype", verdict=v if data else None, confidence=data.get("confidence", 0.3),
        rationale=f"{data.get('comment', 'Phenotype overlap analysis.')} (J={sim:.2f})",
        matched_signals={"jaccard": sim, "matched": list(intersection)},
        raw_model_io=data
    )
    return Verdict(source="phenotype", verdict=ce.verdict, confidence=ce.confidence, notes=ce.rationale, 
                   extras={"jaccard": sim, "explain": asdict(ce)})

def run_literature_channel(case: CaseInput) -> Verdict:
    """Literature Channel: Searches internal Knowledge Base for PubMed evidence."""
    query_tokens = f"{case.hpo_name_list} {case.gene_list} {case.disease}"
    hits = []
    try:
        hits = query_in_KB(query_tokens) or []
    except Exception: pass

    lit_prompt = f"""Summarize PubMed evidence for this case. Return JSON.
    Query: {query_tokens}, Top KB Hits: {json.dumps(hits)}"""

    data = parse_json(justchat(lit_prompt, provider=config['LLM_PROVIDER'])) or {}
    v = str(data.get("verdict", "")).lower() in {"true", "yes"}

    ce = ChannelEvidence(
        source="literature", verdict=v if data else None, confidence=data.get("confidence", 0.3),
        rationale=data.get("note", "Literature review complete."),
        matched_signals={"hits": hits},
        raw_model_io=data
    )
    return Verdict(source="literature", verdict=ce.verdict, confidence=ce.confidence, notes=ce.rationale, extras={"explain": asdict(ce)})



# =========================
# 4. Global Control Loop
# =========================

def verify_with_explanation(
    case: CaseInput, 
) -> Tuple[bool, str, FinalExplanation]:
    
    # 1. Normalization
    case_norm, norm_logs = normalize_case(case)
    
    # 2. Sequential Execution of all Evidence Channels
    kg_v = run_kg_channel(case_norm)
    db_v = run_db_channel(case_norm)
    gene_v = run_gene_channel(case_norm)
    
    # Pass Orphanet results to phenotype channel if available
    orph_res = db_v.extras.get("orphanet_result")
    phen_v = run_phenotype_channel(case_norm, orphanet_result=orph_res)
    lit_v = run_literature_channel(case_norm)

    v_results = [kg_v, db_v, gene_v, phen_v, lit_v]

    # 3. Calculate Scores for the Feature Layer
    features_used = {}
    notes = []
    for v in v_results:
        val = (1.0 if v.yes() else 0.0 if v.no() else 0.5)
        features_used[v.source] = val
        weight = config["THRESHOLDS"]['CHANNEL_WEIGHTS'].get(v.source, 1.0)
        notes.append(f"{v.source}@{weight:.1f} -> {val:.2f}")
    
    weighted_note = "[WEIGHTED] " + " | ".join(notes)

    # 4. Final Adjudication

    # LLM-based final decision logic
    prompt = f"""You are a clinical evidence adjudicator. 
    Case: {case_norm.disease}, Features: {features_used}.
    Return JSON: {{"ok": bool, "Explain": str}}"""
    raw = justchat(prompt, provider=config['LLM_PROVIDER'])
    res = parse_json(raw) or {"ok": False, "Explain": "LLM Adjudication Failed."}
    decision, proba_or_text = res.get("ok", False), res.get("Explain", "")
  

    # 5. Build Explanation Package
    evidences = []
    for v in v_results:
        ex_data = (v.extras or {}).get("explain", {})
        ex_data['score'] = features_used.get(v.source)
        ex_data.setdefault("source", v.source)
        ex_data.setdefault("verdict", v.verdict)
        ex_data.setdefault("confidence", v.confidence)
        ex_data.setdefault("rationale", v.notes)
        evidences.append(ChannelEvidence(**ex_data))

    final_exp = FinalExplanation(
        case_fingerprint=asdict(case_norm),
        channel_evidences=evidences,
        weighted_note=weighted_note,
        features_used=features_used,
        decision=decision,

    )

    summary_text = " | ".join([f"[{v.source}] {v.notes}" for v in v_results] + [weighted_note])
    return decision, summary_text, final_exp

if __name__ == "__main__":
    # Example workflow execution
    test_case = CaseInput(
        case_initial_presentation="Infant with hyperoxaluria...",
        hpo_name_list=["Nephrolithiasis", "Nephrocalcinosis"],
        gene_list=["AGXT"],
        disease="Primary hyperoxaluria type 1"
    )
    
    is_supported, summary, explanation = verify_with_explanation(test_case, use_llm_final=False)
    print(f"Final Decision: {is_supported}")
    print(f"Summary: {summary}")