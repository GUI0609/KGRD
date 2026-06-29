import json
from config_loader import load_config

config = load_config()
import os
import re
import joblib
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
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


def _coerce_verdict(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "yes", "support", "supported", "positive"}:
        return True
    if text in {"false", "no", "oppose", "opposed", "negative", "unsupported"}:
        return False
    return None


def _coerce_confidence(value: Any, default: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    return max(0.0, min(1.0, confidence))


def _pubmed_params(extra: Dict[str, Any]) -> Dict[str, Any]:
    ncbi_config = config.get("NCBI", {})
    params = {"tool": ncbi_config.get("TOOL", "KGRD")}
    if ncbi_config.get("EMAIL"):
        params["email"] = ncbi_config["EMAIL"]
    if ncbi_config.get("API_KEY"):
        params["api_key"] = ncbi_config["API_KEY"]
    params.update(extra)
    return params


def build_pubmed_query(case: CaseInput) -> str:
    def phrase(term: str, field: str = "Title/Abstract") -> str:
        clean = str(term).replace('"', "").strip()
        return f'"{clean}"[{field}]'

    disease = str(case.disease).strip()
    disease_query = f"({phrase(disease)} OR {phrase(disease, 'MeSH Terms')})"
    context_terms = [str(i).strip() for i in (case.gene_list or []) + (case.hpo_name_list or [])[:5] if str(i).strip()]

    if not context_terms:
        return disease_query

    context_query = " OR ".join(phrase(term) for term in context_terms)
    return f"{disease_query} AND ({context_query})"


def query_pubmed_evidence(case: CaseInput) -> List[Dict[str, Any]]:
    retmax = int(config.get("NCBI", {}).get("PUBMED_RETMAX", 10))
    base = config["URLS"]["NCBI_EUTILS"].rstrip("/")
    query = build_pubmed_query(case)

    search_params = _pubmed_params({
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "sort": "relevance",
    })
    search_resp = requests.get(f"{base}/esearch.fcgi", params=search_params, timeout=15)
    search_resp.raise_for_status()
    ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    fetch_params = _pubmed_params({
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml",
    })
    fetch_resp = requests.get(f"{base}/efetch.fcgi", params=fetch_params, timeout=15)
    fetch_resp.raise_for_status()
    root = ET.fromstring(fetch_resp.content)
    hits = []

    for article in root.findall(".//PubmedArticle"):
        title_node = article.find(".//ArticleTitle")
        abstract_parts = []
        for node in article.findall(".//Abstract/AbstractText"):
            text = " ".join(node.itertext()).strip()
            label = node.attrib.get("Label")
            if label and text:
                abstract_parts.append(f"{label}: {text}")
            elif text:
                abstract_parts.append(text)

        article_ids = article.findall(".//PubmedData/ArticleIdList/ArticleId")
        doi = next((node.text or "" for node in article_ids if node.attrib.get("IdType") == "doi"), "")
        hits.append({
            "pmid": article.findtext(".//MedlineCitation/PMID", ""),
            "title": " ".join(title_node.itertext()).strip() if title_node is not None else "",
            "abstract": " ".join(abstract_parts),
            "journal": article.findtext(".//Journal/Title", "") or article.findtext(".//Journal/ISOAbbreviation", ""),
            "pub_year": article.findtext(".//JournalIssue/PubDate/Year", ""),
            "doi": doi,
        })
    return hits


def query_dify_evidence(case: CaseInput) -> List[Dict[str, Any]]:
    query = f"{case.hpo_name_list} {case.gene_list} {case.disease}"
    hits = query_in_KB(query) or []
    return [
        {
            "pmid": item.get("pubmedid", ""),
            "title": "",
            "abstract": item.get("text", ""),
            "journal": "",
            "pub_year": item.get("yearandmonth", ""),
            "doi": "",
            "source": "dify",
        }
        for item in hits
    ]


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
        
        v = _coerce_verdict(data.get("verdict"))
        conf = _coerce_confidence(data.get("confidence"), 0.2)

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
    v = _coerce_verdict(data.get("verdict"))
    
    ce = ChannelEvidence(
        source="db", verdict=v if data else None, confidence=_coerce_confidence(data.get("confidence"), 0.4),
        rationale=data.get("rationale", "DB synthesis complete."),
        matched_signals={"orphacode": orphacode},
        raw_model_io=data
    )
    return Verdict(source="db", verdict=ce.verdict, confidence=ce.confidence, notes=ce.rationale, 
                   extras={"orphacode": orphacode, "orphanet_result": orphanet_result, "explain": asdict(ce)})

def run_gene_channel(case: CaseInput) -> Verdict:
    """Gene-Disease Channel: Validates candidate genes using NCBI Gene data."""
    if not case.gene_list:
        ce = ChannelEvidence(source="gene", verdict=None, confidence=0.0, rationale="No gene provided.")
        return Verdict(source="gene", verdict=None, confidence=0.0, notes="N/A", extras={"explain": asdict(ce)})

    genes = list(dict.fromkeys(str(gene).strip() for gene in case.gene_list if str(gene).strip()))
    gene_records = {}
    for gene in genes:
        try:
            gene_records[gene] = query_ncbi_eutils(term=gene, db='gene') or []
        except Exception as e:
            gene_records[gene] = {"error": repr(e)}

    gene_prompt = f"""Assess whether the candidate gene set supports the proposed disease diagnosis.
    Treat the gene channel as supportive if one or more candidate genes have documented or plausible disease relevance.
    Return STRICT JSON:
    {{"verdict": true/false/null, "confidence": float, "rationale": str, "supporting_genes": [str], "opposing_or_uninformative_genes": [str]}}
    Candidate genes: {genes}
    Disease: {case.disease}
    NCBI Gene records: {json.dumps(gene_records, ensure_ascii=False)}"""

    data = parse_json(justchat(gene_prompt, provider=config['LLM_PROVIDER'])) or {}
    v = _coerce_verdict(data.get("verdict"))

    ce = ChannelEvidence(
        source="gene", verdict=v if data else None, confidence=_coerce_confidence(data.get("confidence"), 0.3),
        rationale=data.get("rationale", "Gene validation finished."),
        matched_signals={
            "genes": genes,
            "gene_records": gene_records,
            "supporting_genes": data.get("supporting_genes", []),
            "opposing_or_uninformative_genes": data.get("opposing_or_uninformative_genes", []),
        },
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
    v = _coerce_verdict(data.get("verdict"))

    ce = ChannelEvidence(
        source="phenotype", verdict=v if data else None, confidence=_coerce_confidence(data.get("confidence"), 0.3),
        rationale=f"{data.get('comment', 'Phenotype overlap analysis.')} (J={sim:.2f})",
        matched_signals={"jaccard": sim, "matched": list(intersection)},
        raw_model_io=data
    )
    return Verdict(source="phenotype", verdict=ce.verdict, confidence=ce.confidence, notes=ce.rationale, 
                   extras={"jaccard": sim, "explain": asdict(ce)})

def run_literature_channel(case: CaseInput) -> Verdict:
    """Literature Channel: Searches the configured literature evidence backend."""
    provider = config.get("LITERATURE_RETRIEVAL", {}).get("PROVIDER", "pubmed").lower()
    query = build_pubmed_query(case) if provider == "pubmed" else f"{case.hpo_name_list} {case.gene_list} {case.disease}"
    try:
        if provider == "dify":
            hits = query_dify_evidence(case)
        elif provider == "pubmed":
            hits = query_pubmed_evidence(case)
        else:
            raise ValueError(f"Unsupported literature retrieval provider: {provider}")
    except Exception as e:
        ce = ChannelEvidence(
            source="literature",
            verdict=None,
            confidence=0.0,
            rationale=f"Literature query failed via {provider}: {e!r}",
            matched_signals={"provider": provider, "query": query, "hits": []},
        )
        return Verdict(source="literature", verdict=None, confidence=0.0, notes=ce.rationale, extras={"explain": asdict(ce)})

    if not hits:
        ce = ChannelEvidence(
            source="literature",
            verdict=None,
            confidence=0.0,
            rationale=f"No literature records were retrieved via {provider}.",
            matched_signals={"provider": provider, "query": query, "hits": []},
        )
        return Verdict(source="literature", verdict=None, confidence=0.0, notes=ce.rationale, extras={"explain": asdict(ce)})

    lit_prompt = f"""Assess whether the retrieved literature records support the proposed diagnosis for this case.
    Use only the literature records shown below. Return STRICT JSON:
    {{"verdict": true/false/null, "confidence": float, "rationale": str, "supporting_pmids": [str]}}
    Retrieval provider: {provider}
    Query: {query}
    Disease: {case.disease}
    Genes: {case.gene_list}
    HPO names: {case.hpo_name_list}
    Literature records: {json.dumps(hits, ensure_ascii=False)}"""

    data = parse_json(justchat(lit_prompt, provider=config['LLM_PROVIDER'])) or {}
    v = _coerce_verdict(data.get("verdict"))

    ce = ChannelEvidence(
        source="literature", verdict=v if data else None, confidence=_coerce_confidence(data.get("confidence"), 0.3),
        rationale=data.get("rationale") or data.get("note", "Literature review complete."),
        matched_signals={"provider": provider, "query": query, "hits": hits, "supporting_pmids": data.get("supporting_pmids", [])},
        raw_model_io=data
    )
    return Verdict(source="literature", verdict=ce.verdict, confidence=ce.confidence, notes=ce.rationale, extras={"explain": asdict(ce)})


def adjudicate_evidence(case: CaseInput, verdicts: List[Verdict]) -> Tuple[bool, Dict[str, Any]]:
    thresholds = config["THRESHOLDS"]
    weights = thresholds["CHANNEL_WEIGHTS"]
    min_confidence = thresholds.get("MIN_CONFIDENCE", 0.6)
    voter_margin = thresholds.get("VOTER", 1.5)

    support_weight = sum(weights.get(v.source, 1.0) * v.confidence for v in verdicts if v.yes())
    oppose_weight = sum(weights.get(v.source, 1.0) * v.confidence for v in verdicts if v.no())
    phenotype = next((v for v in verdicts if v.source == "phenotype"), None)
    gene = next((v for v in verdicts if v.source == "gene"), None)
    literature = next((v for v in verdicts if v.source == "literature"), None)

    gene_and_literature_strongly_positive = (
        gene is not None and gene.yes() and gene.confidence >= min_confidence
        and literature is not None and literature.yes() and literature.confidence >= min_confidence
    )
    phenotype_not_strongly_negative = not (
        phenotype is not None and phenotype.no() and phenotype.confidence >= min_confidence
    ) or gene_and_literature_strongly_positive
    independent_support = any(
        v.source in {"literature", "db", "kg"} and v.yes() and v.confidence >= min_confidence
        for v in verdicts
    )
    gene_dependency_satisfied = not (
        case.gene_list and gene is not None and gene.no() and gene.confidence >= min_confidence
    )
    sufficient_support_margin = support_weight >= oppose_weight + voter_margin

    decision = (
        phenotype_not_strongly_negative
        and sufficient_support_margin
        and independent_support
        and gene_dependency_satisfied
    )
    details = {
        "support_weight": support_weight,
        "oppose_weight": oppose_weight,
        "voter_margin": voter_margin,
        "min_confidence": min_confidence,
        "phenotype_not_strongly_negative": phenotype_not_strongly_negative,
        "gene_and_literature_strongly_positive": gene_and_literature_strongly_positive,
        "independent_support": independent_support,
        "gene_dependency_satisfied": gene_dependency_satisfied,
        "sufficient_support_margin": sufficient_support_margin,
    }
    return decision, details



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
    
    decision, rule_details = adjudicate_evidence(case_norm, v_results)
    weighted_note = "[WEIGHTED] " + " | ".join(notes) + f" | rules={json.dumps(rule_details)}"

    # 4. Final Adjudication
    prompt = f"""You are a clinical evidence adjudicator. Explain the rule-based verifier decision.
    Use only the channel outputs and rule evaluation shown below. Do not override the rule decision.
    Rule decision: {decision}
    Rule evaluation: {json.dumps(rule_details)}
    Channel outputs: {json.dumps([asdict(v) for v in v_results], ensure_ascii=False)}
    Return JSON: {{"Explain": str}}"""
    try:
        raw = justchat(prompt, provider=config['LLM_PROVIDER'])
        res = parse_json(raw) or {}
        proba_or_text = res.get("Explain", "")
    except Exception as e:
        proba_or_text = f"Final explanation unavailable: {e!r}"
  

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

    summary_text = " | ".join([f"[{v.source}] {v.notes}" for v in v_results] + [weighted_note, f"[final] {proba_or_text}"])
    return decision, summary_text, final_exp


def verify(case: CaseInput) -> Dict[str, Any]:
    decision, summary, explanation = verify_with_explanation(case)
    return {
        "ok": decision,
        "explanation": summary,
        "details": asdict(explanation),
    }

if __name__ == "__main__":
    # Example workflow execution
    test_case = CaseInput(
        case_initial_presentation="Infant with hyperoxaluria...",
        hpo_name_list=["Nephrolithiasis", "Nephrocalcinosis"],
        gene_list=["AGXT"],
        disease="Primary hyperoxaluria type 1"
    )
    
    is_supported, summary, explanation = verify_with_explanation(test_case)
    print(f"Final Decision: {is_supported}")
    print(f"Summary: {summary}")
