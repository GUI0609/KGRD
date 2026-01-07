import json
with open('PATH/TO/config.json', 'r') as f:
    config = json.load(f)
# ===============
# JSON and General Utilities
# ===============
import re

import time
import ast
from functools import wraps
from autogen_agentchat.base import Response

from typing import Any, Dict, List, Literal

from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    HandoffMessage,
    MultiModalMessage,
    StopMessage,
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from autogen_core import FunctionCall, Image
from autogen_core.models import FunctionExecutionResult
# Regex for quote normalization
_SINGLE_QUOTED_STRING_RE = re.compile(r"'([^'\\]*(?:\\.[^'\\]*)*)'")
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
def to_valid_identifier(name: str) -> str:
    name = re.sub(r'\W|^(?=\d)', '_', name)
    return name or "_agent"
def normalize_quotes(text: str) -> str:
    """Standardizes quotes for JSON compatibility."""
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    if '"' in text and text.count("'") == 0:
        return text

    def repl(m: re.Match) -> str:
        inner = m.group(1).replace('"', '\\"')
        return f'"{inner}"'
    return _SINGLE_QUOTED_STRING_RE.sub(repl, text)

def fix_trailing_commas(text: str) -> str:
    """Removes trailing commas in JSON objects/arrays."""
    return _TRAILING_COMMA_RE.sub(r"\1", text)

def parse_json(text: str):
    """
    Robust JSON parser attempting multiple recovery strategies:
    1. Markdown fenced blocks
    2. LaTeX boxed/json environments
    3. Balanced brace scanning
    """
    def _try_load(s: str):
        s = s.strip()
        try: return json.loads(s)
        except: pass
        try:
            import json5
            return json5.loads(s)
        except: pass
        try: return ast.literal_eval(s)
        except: return None

    # 1. Search in Markdown code blocks
    for pat in (r"```json\s*(.*?)\s*```", r"```JSON\s*(.*?)\s*```", r"```\s*(.*?)\s*```"):
        m = re.search(pat, text, re.DOTALL)
        if m:
            obj = _try_load(m.group(1))
            if obj is not None: return obj

    # 2. Search in LaTeX environments
    m = re.search(r"\\begin\s*\{\s*json\s*\}([\s\S]*?)\\end\s*\{\s*json\s*\}", text, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        obj = _try_load(candidate) or _try_load(fix_trailing_commas(normalize_quotes(candidate)))
        if obj is not None: return obj

    # 3. Balanced brace scan
    def _scan(s: str):
        opens, start, in_str, esc = [], None, False, False
        for i, ch in enumerate(s):
            if start is None:
                if ch in "{[":
                    start, opens = i, [ch]
                continue
            if in_str:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_str = False
            else:
                if ch == '"': in_str = True
                elif ch in "{[": opens.append(ch)
                elif ch in "}]":
                    if not opens: start = None; continue
                    opener = opens.pop()
                    if (opener == "{" and ch == "}") or (opener == "[" and ch == "]"):
                        if not opens:
                            obj = _try_load(s[start:i+1])
                            if obj is not None: return obj
                    else: start, opens = None, []
        return None

    obj = _scan(text)
    if obj: return obj
    
    if text.lstrip().startswith(("{", "[")):
        obj = _try_load(text)
        if obj: return obj

    raise json.JSONDecodeError("Failed to parse JSON from text", text, 0)

def simple_retry(max_attempts=3, delay=1):
    """Generic retry decorator for network or LLM calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
                    else:
                        raise e
        return wrapper
    return decorator

def content_str(reply: Response | str) -> str:
    """Extracts content string from various response types."""
    if isinstance(reply, str): return reply
    if hasattr(reply, "chat_message"): return reply.chat_message.content
    return str(reply)



# ===============
# Medical Knowledge and API Tools
# ===============

import requests
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import pandas as pd
import math

# 加载 SAPBERT 模型（只加载一次）
SapBERTmodel_name = config['PATHS']['SAPBERT']
SapBERTtokenizer = AutoTokenizer.from_pretrained(SapBERTmodel_name)
SapBERTmodel = AutoModel.from_pretrained(SapBERTmodel_name)
SapBERTmodel.eval()

# 使用 GPU（如果可用）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SapBERTmodel = SapBERTmodel.to(device)

WORD_LIST_PATH = config['PATHS']['WORD_LIST_PATH']
WORD_LIST = pd.read_csv(WORD_LIST_PATH)['n.entity_id'].tolist()
WORD_LIST = [str(w) for w in WORD_LIST]
LOWER_WORD_MAP = {w.lower(): w for w in WORD_LIST}

WORD_LIST_EMBEDDINGS = None

def encode_for_SapBERT(texts, batch_size=32, max_length=128):
    if not texts:
        return torch.empty(0, 768)
    
    if isinstance(texts, str):
        texts = [texts]
    
    texts = [str(t) for t in texts]
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        with torch.no_grad():
            inputs = SapBERTtokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            
            outputs = SapBERTmodel(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings.cpu())
            
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0, 768)

def precompute_word_embeddings():
    global WORD_LIST_EMBEDDINGS
    
    if WORD_LIST_EMBEDDINGS is not None:
        return
    
    all_embeddings = []
    batch_size = 64
    total = len(WORD_LIST)
    
    for i in range(0, total, batch_size):
        batch = WORD_LIST[i:i + batch_size]
        emb = encode_for_SapBERT(batch, batch_size=batch_size)
        all_embeddings.append(emb)
    
    WORD_LIST_EMBEDDINGS = torch.cat(all_embeddings, dim=0)

import requests
#用于patientlikeme
def gene_name2ensembl(genes, species='homo_sapiens'):

    if isinstance(genes, str):
        genes = [genes]

    result = {}
    for gene in genes:
        url = f'https://rest.ensembl.org/lookup/symbol/{species}/{gene}?content-type=application/json'
        r = requests.get(url, headers={"Content-Type": "application/json"})
        if r.ok:
            data = r.json()
            result[gene] = data.get('id')
        else:
            result[gene] = None  # 查询失败时返回 None
    return result
def load_hp_obo(file_path):

    hp_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        current_id = None
        for line in f:
            line = line.strip()
            if line.startswith('id:'):
                current_id = line.split('id:')[1].strip()
            elif line.startswith('name:') and current_id:
                name = line.split('name:')[1].strip()
                hp_dict[current_id] = name
                current_id = None  # 重置以防干扰下一个条目
    return hp_dict
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re

hp_file = "/share/gguilin/rd-project/KGRD/src/KGRD_framework/kg/hp.obo"
hp_dict = load_hp_obo(hp_file)
def hpo_to_name(hp_code, hp_dict = hp_dict):

    return hp_dict.get(hp_code, "")

def hpname_to_hpo(name, hp_dict=hp_dict, ignore_case=True):

    if ignore_case:
        name_to_code = {v.lower(): k for k, v in hp_dict.items()}
        return name_to_code.get(name.lower(), "")
    else:
        name_to_code = {v: k for k, v in hp_dict.items()}
        return name_to_code.get(name, "")


import requests
import re
from bs4 import BeautifulSoup

def query_kg(data):
    """
    Generic POST helper for local knowledge graph services.
    """
    try:
        # 确保 config 变量在全局作用域内可用
        url = config['URLS']['KNOWLEDGE_GRAPH']
        resp = requests.post(url, json=data)
        return resp.json() if resp.status_code == 200 else {"error": resp.status_code}
    except Exception as e:
        return {"error": str(e)}


def query_ncbi_eutils(term, db, retmax=5):
    """
    Interface for NCBI Entrez E-Utilities.
    """
    base = config['URLS']['NCBI_EUTILS']
    
    # 1. Search: 获取 ID 列表
    s_params = {"db": db, "term": term, "retmax": retmax, "retmode": "json"}
    s_resp = requests.get(f"{base}/esearch.fcgi", params=s_params).json()
    ids = s_resp.get("esearchresult", {}).get("idlist", [])
    
    if not ids:
        return []
    
    # 2. Summary: 根据 ID 获取详细信息
    sum_params = {"db": db, "id": ",".join(ids), "retmode": "json"}
    sum_resp = requests.get(f"{base}/esummary.fcgi", params=sum_params).json()
    results = sum_resp.get("result", {})
    
    return [
        f"{results[uid].get('title', uid)}: {results[uid].get('pubdate', '')}" 
        for uid in results.get("uids", [])
    ]


def fetch_variant_interpretation(chr_num, pos, ref, alt, build="hg19"):
    """
    Scrapes pathogenicity info from wglab.org.
    """
    url = "http://wintervar.wglab.org/results.pos.php"
    params = {
        "queryType": "position", 
        "chr": chr_num, 
        "pos": pos, 
        "ref": ref, 
        "alt": alt, 
        "build": build
    }
    
    resp = requests.get(url, params=params)
    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text(separator="\n")
    
    fields = {
        "Build": r"build:\s*(\w+)", 
        "Gene": r"Gene:\s*(\w+)", 
        "Interpretation": r"clinical interpretation is\s*:\s*(.+?)\s*,"
    }
    
    info = {}
    for key, pattern in fields.items():
        match = re.search(pattern, text)
        if match:
            # 注意：Interpretation 的正则使用了 group(1) 
            # 如果匹配项较多，请确保逻辑与原类一致
            info[key] = match.group(1)
            
    return info


import requests

class OrphadataAPI:
    BASE_URL = "https://api.orphadata.com"

    @staticmethod
    def get_icd10s(lang="en"):
        url = f"{OrphadataAPI.BASE_URL}/rd-cross-referencing/icd-10s?lang={lang}"
        return requests.get(url).json()

    @staticmethod
    def get_icd10_by_code(icd, lang="en"):
        url = f"{OrphadataAPI.BASE_URL}/rd-cross-referencing/icd-10s/{icd}?lang={lang}"
        return requests.get(url).json()

    @staticmethod
    def get_icd11s(lang="en"):
        url = f"{OrphadataAPI.BASE_URL}/rd-cross-referencing/icd-11s?lang={lang}"
        return requests.get(url).json()

    @staticmethod
    def get_icd11_by_code(icd, lang="en"):
        url = f"{OrphadataAPI.BASE_URL}/rd-cross-referencing/icd-11s/{icd}?lang={lang}"
        return requests.get(url).json()

    @staticmethod
    def get_omims(lang="en"):
        url = f"{OrphadataAPI.BASE_URL}/rd-cross-referencing/omims?lang={lang}"
        return requests.get(url).json()

    @staticmethod
    def get_omim_by_code(omim, lang="en"):
        url = f"{OrphadataAPI.BASE_URL}/rd-cross-referencing/omims/{omim}?lang={lang}"
        return requests.get(url).json()

    @staticmethod
    def get_orphacodes(lang="en"):
        url = f"{OrphadataAPI.BASE_URL}/rd-cross-referencing/orphacodes?lang={lang}"
        return requests.get(url).json()

    @staticmethod
    def get_orphacode_by_name(name, lang="en"):
        url = f"{OrphadataAPI.BASE_URL}/rd-cross-referencing/orphacodes/names/{name}?lang={lang}"
        return requests.get(url).json()

    @staticmethod
    def get_orphacode_detail(orphacode, lang="en"):
        url = f"{OrphadataAPI.BASE_URL}/rd-cross-referencing/orphacodes/{orphacode}?lang={lang}"
        return requests.get(url).json()

    @staticmethod
    def get_classifications():
        url = f"{OrphadataAPI.BASE_URL}/rd-classification/hchids"
        return requests.get(url).json()

    @staticmethod
    def get_classification_by_hchid(hchid):
        url = f"{OrphadataAPI.BASE_URL}/rd-classification/hchids/{hchid}"
        return requests.get(url).json()

    @staticmethod
    def get_classification_orphacodes(hchid):
        url = f"{OrphadataAPI.BASE_URL}/rd-classification/hchids/{hchid}/orphacodes"
        return requests.get(url).json()

    @staticmethod
    def get_classification_orphacode_terms():
        url = f"{OrphadataAPI.BASE_URL}/rd-classification/orphacodes"
        return requests.get(url).json()

    @staticmethod
    def get_classification_orphacode_hchids(orphacode):
        url = f"{OrphadataAPI.BASE_URL}/rd-classification/orphacodes/{orphacode}/hchids"
        return requests.get(url).json()

    @staticmethod
    def get_classification_orphacode_hchid(orphacode, hchid):
        url = f"{OrphadataAPI.BASE_URL}/rd-classification/orphacodes/{orphacode}/hchids/{hchid}"
        return requests.get(url).json()

    @staticmethod
    def get_hpoids(lang="en"):
        url = f"{OrphadataAPI.BASE_URL}/rd-phenotypes/hpoids?lang={lang}"
        return requests.get(url).json()

    @staticmethod
    def get_phenotype_by_hpoids(hpoids, lang="en"):
        url = f"{OrphadataAPI.BASE_URL}/rd-phenotypes/hpoids/{hpoids}?lang={lang}"
        return requests.get(url).json()

    @staticmethod
    def get_phenotype_orphacodes(lang="en"):
        url = f"{OrphadataAPI.BASE_URL}/rd-phenotypes/orphacodes?lang={lang}"
        return requests.get(url).json()

    @staticmethod
    def get_phenotype_by_orphacode(orphacode, lang="en"):
        url = f"{OrphadataAPI.BASE_URL}/rd-phenotypes/orphacodes/{orphacode}?lang={lang}"
        return requests.get(url).json()

    @staticmethod
    def get_genes():
        url = f"{OrphadataAPI.BASE_URL}/rd-associated-genes/genes"
        return requests.get(url).json()

    @staticmethod
    def get_genes_by_name(name):
        url = f"{OrphadataAPI.BASE_URL}/rd-associated-genes/genes/names/{name}"
        return requests.get(url).json()

    @staticmethod
    def get_genes_by_symbol(symbol):
        url = f"{OrphadataAPI.BASE_URL}/rd-associated-genes/genes/symbols/{symbol}"
        return requests.get(url).json()

    @staticmethod
    def get_gene_orphacodes():
        url = f"{OrphadataAPI.BASE_URL}/rd-associated-genes/orphacodes"
        return requests.get(url).json()

    @staticmethod
    def get_gene_by_orphacode(orphacode):
        url = f"{OrphadataAPI.BASE_URL}/rd-associated-genes/orphacodes/{orphacode}"
        return requests.get(url).json()

    @staticmethod
    def get_medical_specialties_orphacodes():
        url = f"{OrphadataAPI.BASE_URL}/rd-medical-specialties/orphacodes"
        return requests.get(url).json()

    @staticmethod
    def get_preferential_parent_by_orphacode(orphacode):
        url = f"{OrphadataAPI.BASE_URL}/rd-medical-specialties/orphacodes/{orphacode}"
        return requests.get(url).json()

    @staticmethod
    def get_medical_specialties_parents():
        url = f"{OrphadataAPI.BASE_URL}/rd-medical-specialties/parents"
        return requests.get(url).json()

    @staticmethod
    def get_medical_specialties_by_parentcode(parentcode):
        url = f"{OrphadataAPI.BASE_URL}/rd-medical-specialties/parents/{parentcode}"
        return requests.get(url).json()

    @staticmethod
    def get_epidemiology_orphacodes(lang="en"):
        url = f"{OrphadataAPI.BASE_URL}/rd-epidemiology/orphacodes?lang={lang}"
        return requests.get(url).json()

    @staticmethod
    def get_epidemiology_by_orphacode(orphacode, lang="en"):
        url = f"{OrphadataAPI.BASE_URL}/rd-epidemiology/orphacodes/{orphacode}?lang={lang}"
        return requests.get(url).json()

    @staticmethod
    def get_natural_history_orphacodes(lang="en"):
        url = f"{OrphadataAPI.BASE_URL}/rd-natural_history/orphacodes?lang={lang}"
        return requests.get(url).json()

    @staticmethod
    def get_natural_history_by_orphacode(orphacode, lang="en"):
        url = f"{OrphadataAPI.BASE_URL}/rd-natural_history/orphacodes/{orphacode}?lang={lang}"
        return requests.get(url).json()
import time
import requests
from typing import List, Dict, Optional, Union, Any, Tuple

class PhenoBrain:
    BASE_URL: str = "http://www.phenobrain.cs.tsinghua.edu.cn"

    def __init__(self) -> None:
        self.model = "Ensemble"
        self.topk = 100
        self.poll_interval = 1

    def predict_disease(self, hpo_list: List[str]) -> List[Dict[str, Any]]:
        predict_url = f"{self.BASE_URL}/predict"
        params = [("model", self.model), ("topk", self.topk)]
        params += [("hpoList[]", hpo) for hpo in hpo_list]

        response = requests.get(predict_url, params=params)
        response.raise_for_status()
        task_id = response.json().get("TASK_ID")

        if not task_id:
            raise RuntimeError("Failed to get TASK_ID, please check request parameters.")

        query_url = f"{self.BASE_URL}/query-predict-result"
        while True:
            query_response = requests.get(query_url, params={"taskId": task_id})
            query_response.raise_for_status()
            result_data = query_response.json()
            if result_data.get("state") == "SUCCESS":
                return result_data.get("result", [])
            time.sleep(self.poll_interval)

    def get_disease_detail(self, dis_code: str, pa_hpo_list: Optional[List[str]] = None) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/disease-detail"
        params: List[Any] = [("disCode", dis_code)]
        if pa_hpo_list:
            params += [("paHpoList[]", hpo) for hpo in pa_hpo_list]

        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def predict_disease_with_details(self, hpo_list: List[str]) -> List[str]:
        basic_results = self.predict_disease(hpo_list)
        detailed_results = []
        for disease in basic_results:
            code = disease.get("CODE")
            if code:
                detail = self.get_disease_detail(code, pa_hpo_list=hpo_list)
                detail["SCORE"] = disease.get("SCORE")
                detailed_results.append(detail)
        return [disease.get('ENG_NAME', '') for disease in detailed_results]

    def get_hpo_tree(self) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/hpo-tree-init"
        response = requests.get(url)
        return response.json()

    def get_hpo_child(self, hpo_code: str) -> List[Dict[str, Any]]:
        url = f"{self.BASE_URL}/hpo-child"
        params = {"hpo": hpo_code}
        response = requests.get(url, params=params)
        return response.json()

    def get_hpo_child_many(self, hpo_list: List[str]) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/hpo-child-many"
        params = [(f"hpoList[]", hpo) for hpo in hpo_list]
        response = requests.get(url, params=params)
        return response.json()

    def get_hpo_detail(self, hpo_code: str, projections: Optional[List[str]] = None) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/hpo-detail"
        params: List[Any] = [("hpo", hpo_code)]
        if projections:
            for proj in projections:
                params.append(("projection[]", proj))
        response = requests.get(url, params=params)
        return response.json()

    def extract_hpo(self, text: str, method: str = "HPO/CHPO", threshold: str = "") -> Optional[Dict[str, Any]]:
        # Step 1: Submit Task
        submit_url = f"{self.BASE_URL}/extract-hpo"
        headers = {'Content-Type': 'application/json'}
        payload = {
            "text": text,
            "method": method,
            "threshold": threshold
        }

        print(f"1. Submitting task to: {submit_url}")
        try:
            response = requests.post(submit_url, json=payload, headers=headers)
            response.raise_for_status()
            
            submit_data = response.json()
            task_id = submit_data.get("TASK_ID")
            
            if not task_id:
                raise ValueError(f"Failed to get TASK_ID, server returned: {submit_data}")
            
            print(f"   Task submitted successfully, TASK_ID: {task_id}")

        except Exception as e:
            print(f"   Task submission failed: {e}")
            return None

        # Step 2: Poll Results
        query_url = f"{self.BASE_URL}/query-extract-hpo-result"
        max_retries = 30
        wait_seconds = 2
        
        print("2. Polling for results...")
        for i in range(max_retries):
            try:
                query_params = {"taskId": task_id}
                query_resp = requests.get(query_url, params=query_params)
                query_resp.raise_for_status()
                
                result_data = query_resp.json()
                state = result_data.get("state")
                
                if state == "SUCCESS":
                    print("   State: SUCCESS! Result retrieved.")
                    return result_data.get("result")
                
                elif state in ["PROCESS_TEXT", "EXTRACT_HPO"]:
                    print(f"   State: {state}, retrying in {wait_seconds}s... ({i+1}/{max_retries})")
                    time.sleep(wait_seconds)
                
                else:
                    print(f"   Unknown state: {state}")
                    time.sleep(wait_seconds)

            except Exception as e:
                print(f"   Error during polling: {e}")
                time.sleep(wait_seconds)
        
        raise TimeoutError("Request timed out, failed to get SUCCESS state.")

    def search_hpo(self, query_text: str) -> List[Dict[str, Any]]:
        url = f"{self.BASE_URL}/search-hpo"
        params = {"query": query_text}
        response = requests.get(url, params=params)
        return response.json()

    def search_dis(self, query_text: str) -> List[Dict[str, Any]]:
        url = f"{self.BASE_URL}/search-dis"
        params = {"query": query_text}
        response = requests.get(url, params=params)
        return response.json()



def g2d(genes:list):
    try:
        url = "http://localhost:8092/predict"
        data = {"genes": genes}
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"GNN predict failed with status {response.status_code}"}
    except Exception as e:
        return {"error": f"GNN predict exception: {str(e)}"}


from bs4 import BeautifulSoup
import re

def query_variant_Interpretation_Pathogenicity(chr_num, pos, ref, alt, build="hg19"):
    url = "http://wintervar.wglab.org/results.pos.php"
    params = {
        "queryType": "position",
        "chr": chr_num,
        "pos": pos,
        "ref": ref,
        "alt": alt,
        "build": build
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")

    text = soup.get_text(separator="\n")
    info = {}

    build_match = re.search(r"build:\s*(\w+)", text)
    chr_match = re.search(r"Chr:\s*(\w+)", text)
    pos_match = re.search(r"Pos:\s*(\d+)", text)
    ref_match = re.search(r"Ref:\s*(\w+)", text)
    alt_match = re.search(r"Alt:\s*(\w+)", text)
    gene_match = re.search(r"Gene:\s*(\w+)", text)
    interp_match = re.search(r"The automated clinical interpretation is\s*:\s*(.+?)\s*,", text)

    if build_match: info["Build"] = build_match.group(1)
    if chr_match: info["Chromosome"] = chr_match.group(1)
    if pos_match: info["Position"] = pos_match.group(1)
    if ref_match: info["Ref"] = ref_match.group(1)
    if alt_match: info["Alt"] = alt_match.group(1)
    if gene_match: info["Gene"] = gene_match.group(1)
    if interp_match: info["Interpretation"] = interp_match.group(1)

    return info

def get_hpo_detail(hpo_code, projections=["ENG_NAME","ENG_DEF", "SYNONYM", "REL_DIS"]):
    BASE_URL = "http://www.phenobrain.cs.tsinghua.edu.cn"
    url = f"{BASE_URL}/hpo-detail"
    params = [("hpo", hpo_code)]
    if projections:
        for proj in projections:
            params.append(("projection[]", proj))
    response = requests.get(url, params=params)
    return response.json()


def sapbert_d_patient(hpo_name_list: Optional[List[str]] = None):
    BASE_URL = "http://localhost:6006"
    url = f"{BASE_URL}/sapbert_match_patients"
    payload = {
            "batch_id_hpo_dict": {'test_patient_000':hpo_name_list}
        }
    print(f"🔍 Testing: {url}")
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()  # 不要使用 json.dumps
        return result
    else:
        print("❌ Error:", response.status_code, response.text)

import re
import json
config = json.load(open("PATH/TO/config.json"))
def query_in_KB(text_input: str) -> dict:
    # API 配置
    API_URL = "http://0.0.0.0/v1/workflows/run"
    API_KEY = config['API_KEYS']['DIFI']

    # 构建请求
    payload = {
        "inputs": {
            "text": text_input
        },
        "response_mode": "blocking",
        "user": "user-001"
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }


    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise RuntimeError(f"error: {response.status_code}, content:{response.text}")

    data = response.json()
    output = data.get("data", {}).get("outputs", {}).get("output")

    result = []

    month_pattern = r"(19|20)\d{2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    
    for idx, item in enumerate(output):
        full_text = item.get("content", "").strip()
        match = re.search(month_pattern, full_text)
        if match:
            start_index = match.start()
            prefix = full_text[:start_index].strip().split()
            yearandmonth = match.group()
            main_text = full_text[start_index + len(yearandmonth):].strip()
            if len(prefix) >= 2:
                result.append({
                    "index": prefix[0],
                    "pubmedid": prefix[1],
                    "yearandmonth": yearandmonth,
                    "text": main_text
                })

    return result

def call_api_requests(method, text, api_key=None):
    '''Method:["actree", "scispacy", "gpt"]'''
    url = f"http://localhost:5010/api/search/{method}"
    

    if method == 'gpt':
        data = {
            "text": text,
            "openaiKey": api_key,
        }
    else:
        data = {
            "text": text,
        }
    

    response = requests.post(url, json=data)
    

    if response.status_code == 200:
        result = response.json()
        hpo_list = list(set([i[3].get('id') for i in result]))
        return hpo_list
    else:
        print("Error:", response.status_code, response.text)
# ===============
# Model and Embedding Engine
# ===============

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel

class SapBertEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SapBertEngine, cls).__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        print("Initializing SapBERT Engine...")
        self.tokenizer = AutoTokenizer.from_pretrained(config['PATHS']['SAPBERT'])
        self.model = AutoModel.from_pretrained(config['PATHS']['SAPBERT']).to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        
        # Load dictionary
        df = pd.read_csv(config['PATHS']['ENTITY_ID_CSV'])
        self.word_list = [str(w) for w in df['n.entity_id'].tolist()]
        self.lower_map = {w.lower(): w for w in self.word_list}
        self.embeddings = None

    def encode(self, texts, batch_size=32):
        if isinstance(texts, str): texts = [texts]
        all_embs = []
        device = next(self.model.parameters()).device
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                out = self.model(**inputs)
                all_embs.append(out.last_hidden_state[:, 0, :].cpu())
        return torch.cat(all_embs, dim=0)

    def precompute_dictionary(self):
        if self.embeddings is not None: return
        print(f"Precomputing embeddings for {len(self.word_list)} entities...")
        self.embeddings = self.encode(self.word_list, batch_size=64)

    def find_best_match(self, query):
        if not query: return None
        query = query.strip()
        if query.lower() in self.lower_map: return self.lower_map[query.lower()]
        
        self.precompute_dictionary()
        q_emb = self.encode([query])[0]
        scores = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), self.embeddings)
        best_idx = torch.argmax(scores).item()
        return self.word_list[best_idx]


# ===============
# Agent Orchestration and LLM Client
# ===============
from openai import OpenAI
import ollama


def justchat(prompt, provider='deepseek', model="deepseek-chat", temperature=1.0):
    """Unified chat interface for various LLM providers."""
    if provider == 'deepseek':
        client = OpenAI(api_key=config['API_KEYS']['DEEPSEEK'], base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=temperature)
        return resp.choices[0].message.content
    elif provider == 'ollama':
        resp = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}], options={'temperature': temperature})
        return resp['message']['content']
    # Add other providers similarly...
    return "Error: Unsupported provider"

@simple_retry(max_attempts=3)
def retry_llm_selector(last_message, doc_msg_counts, docs_names, verifier, supervisor_name, agents, tools_name):
    """Strategic orchestrator to decide the next speaker in a medical consultation."""
    
    rule_v = f"Select **Verifier** ({verifier}) for methodological review." if verifier else "Proceed to summary."
    tools_line = f"- Tools: {', '.join(tools_name)}" if tools_name else ""

    prompt = f"""
    You are a medical conference orchestrator. Decide who speaks next based on:
    - History: {last_message.content}
    - Turn Stats: {doc_msg_counts}
    - Doctors: {', '.join(docs_names)}
    - Verifier: {verifier}
    - Supervisor: {supervisor_name}
    {tools_line}

    Selection Rules:
    1. Need reasoning? Select relevant Doctor with fewer turns.
    2. Consensus reached? {rule_v}
    3. Final synthesis? Select {supervisor_name}.
    
    Return ONLY the exact name string.
    """
    
    response = justchat(prompt).strip()
    valid_names = [a.name for a in agents]
    if response in valid_names and response != last_message.source:
        return response
    
    # Fallback logic
    return docs_names[0]

# ============================================================
#  KG
# ============================================================
def find_best_match_entity(query):
    """
    Finds the entity that best matches the query term.
    
    Optimization points:
    1. Word list is loaded only once (global variable).
    2. Word list embeddings are precomputed once.
    3. Each query only requires 1 embedding calculation (Extremely fast!).
    
    Args:
        query: The search term (string).
    
    Returns:
        str: The name of the best matching entity.
    """
    global WORD_LIST_EMBEDDINGS
    
    # Ensure precomputation is completed
    if WORD_LIST_EMBEDDINGS is None:
        precompute_word_embeddings()
    
    # Handle input types
    if isinstance(query, list):
        query = str(query)
    
    if not query or not str(query).strip():
        return None
    
    query = str(query).strip()
    
    # 1. Try exact matching first (case-insensitive) - Fastest, O(1)
    if query.lower() in LOWER_WORD_MAP:
        return LOWER_WORD_MAP[query.lower()]
    
    # 2. Vector similarity matching - Calculate only 1 embedding for the query
    query_embedding = encode_for_SapBERT([query], batch_size=1)[0]  # [768]
    
    # Calculate cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        query_embedding.unsqueeze(0),  # [1, 768]
        WORD_LIST_EMBEDDINGS           # [N, 768]
    )
    
    # Retrieve the most similar result
    topk = torch.topk(cos_sim, k=1)
    best_idx = topk.indices[0].item()
    best_score = topk.values[0].item()
    
    # Optional: Print warning for low-confidence matches
    if best_score < 0.5:
        print(f"  ⚠️ '{query}' → '{WORD_LIST[best_idx]}' (Similarity: {best_score:.3f})")
    
    return WORD_LIST[best_idx]

# ============================================================
# 5. [New] Subgraph Processing Function - Modify based on your actual format
# ============================================================
def process_subgraph(subgraph_result: list, 
                     query_entities: dict,
                     max_nodes: int = 30,
                     max_rels: int = 50,
                     output_format: str = "json"):
    """
    Processes the subgraph: prunes the quantity and converts the format.
    
    Args:
        subgraph_result: Original subgraph list [{"nodes": [...], "rels": [...]}]
        query_entities: Dictionary of query entities
        max_nodes: Maximum number of nodes to return
        max_rels: Maximum number of relations to return
        output_format: "json", "text", or "summary"
    
    Returns:
        The processed subgraph in the specified format.
    """
    # Handle empty results
    if not subgraph_result:
        return {"nodes": [], "rels": [], "message": "No relevant subgraph found"}
    
    # Take the first subgraph if it's a list
    if isinstance(subgraph_result, list):
        subgraph = subgraph_result[0] if subgraph_result else {"nodes": [], "rels": []}
    else:
        subgraph = subgraph_result
    
    nodes = subgraph.get("nodes", [])
    rels = subgraph.get("rels", [])
    
    if not nodes and not rels:
        return {"nodes": [], "rels": [], "message": "No relevant subgraph found"}
    
    # Flatten the set of query entities
    query_entity_set = set()
    if isinstance(query_entities, dict):
        for v in query_entities.values():
            if v:
                query_entity_set.update(v)
    
    # -------- Calculate Node Importance --------
    # Count the degree of each node (how many relations it participates in)
    node_degree = {}
    for rel in rels:
        if len(rel) >= 3:
            source_id = rel[0].get("entity_id") if isinstance(rel[0], dict) else rel[0]
            target_id = rel[2].get("entity_id") if isinstance(rel[2], dict) else rel[2]
            node_degree[source_id] = node_degree.get(source_id, 0) + 1
            node_degree[target_id] = node_degree.get(target_id, 0) + 1
    
    def node_priority(node):
        """Calculates node priority score"""
        entity_id = node.get("entity_id", "")
        score = 0
        # Query entities get the highest priority
        if entity_id in query_entity_set:
            score += 10000
        # Higher degree implies higher importance
        score += node_degree.get(entity_id, 0) * 10
        return score
    
    # Sort by priority and prune nodes
    sorted_nodes = sorted(nodes, key=node_priority, reverse=True)
    limited_nodes = sorted_nodes[:max_nodes]
    limited_node_ids = set(n.get("entity_id") for n in limited_nodes)
    
    # -------- Prune Relations --------
    def rel_priority(rel):
        """Calculates relation priority score"""
        if len(rel) < 3:
            return 0
        source_id = rel[0].get("entity_id") if isinstance(rel[0], dict) else rel[0]
        target_id = rel[2].get("entity_id") if isinstance(rel[2], dict) else rel[2]
        score = 0
        if source_id in query_entity_set:
            score += 100
        if target_id in query_entity_set:
            score += 100
        return score
    
    # Only keep relations where both ends are within the limited nodes
    relevant_rels = []
    for rel in rels:
        if len(rel) >= 3:
            source_id = rel[0].get("entity_id") if isinstance(rel[0], dict) else rel[0]
            target_id = rel[2].get("entity_id") if isinstance(rel[2], dict) else rel[2]
            if source_id in limited_node_ids and target_id in limited_node_ids:
                relevant_rels.append(rel)
    
    # Sort by priority and prune relations
    sorted_rels = sorted(relevant_rels, key=rel_priority, reverse=True)
    limited_rels = sorted_rels[:max_rels]
    
    # Nodes: Extract the entity_id values directly
    simplified_nodes = [n.get("entity_id") for n in limited_nodes]
    
    # Relations: Map to [source_id, rel_type, target_id]
    simplified_rels = []
    for rel in limited_rels:
        if len(rel) >= 3:
            source_id = rel[0].get("entity_id") if isinstance(rel[0], dict) else rel[0]
            rel_type = rel[1].get("type") if isinstance(rel[1], dict) else rel[1]
            target_id = rel[2].get("entity_id") if isinstance(rel[2], dict) else rel[2]
            simplified_rels.append([source_id, rel_type, target_id])
    
    if output_format == "text":
        return format_as_text(simplified_nodes, simplified_rels, query_entity_set, original_stats)
    else:  # Return as JSON
        return {
            "nodes": simplified_nodes,
            "rels": simplified_rels
        }


def format_as_text(nodes, rels, query_entities, stats):
    """
    Converts graph data to an LLM-friendly text format.
    """
    if not rels:
        node_names = [n.get("name", n.get("entity_id", "")) for n in nodes[:10]]
        return f"Found {stats['original_nodes']} relevant entities: {', '.join(node_names)}..."
    
    lines = []
    lines.append(f"Knowledge Graph Info (Showing {stats['returned_rels']}/{stats['original_rels']} relations):\n")
    
    # Group by relation type
    relation_groups = {}
    for rel in rels:
        if len(rel) >= 3:
            rel_type = rel[1] if isinstance(rel[1], str) else "related"
            if rel_type not in relation_groups:
                relation_groups[rel_type] = []
            relation_groups[rel_type].append(rel)
    
    # Output each relation type
    for rel_type, type_rels in relation_groups.items():
        lines.append(f"\n[{rel_type}]")
        for rel in type_rels[:15]:  # Limit to 15 entries per relation type
            source = rel[0]
            target = rel[2]
            
            source_name = source.get("name", source.get("entity_id", "?")) if isinstance(source, dict) else str(source)
            target_name = target.get("name", target.get("entity_id", "?")) if isinstance(target, dict) else str(target)
            
            # Mark query entities with brackets
            source_id = source.get("entity_id", "") if isinstance(source, dict) else source
            target_id = target.get("entity_id", "") if isinstance(target, dict) else target
            
            if source_id in query_entities:
                source_name = f"[{source_name}]"
            if target_id in query_entities:
                target_name = f"[{target_name}]"
            
            lines.append(f"  • {source_name} → {target_name}")
    
    return "\n".join(lines)


# API Wrapper functions for local services
def query_gene_to_disease(gene_ids: list):
    try:
        url = "http://localhost:8194/query_gene_to_disease"
        data = {"gene_ids": gene_ids}
        response = requests.post(url, json=data)
        return [i['disease'] for i in response.json()] if response.status_code == 200 else {"error": response.status_code}
    except Exception as e:
        return {"error": str(e)}

def query_phenotype_to_disease(phenotype_ids: list):
    try:
        url = "http://localhost:8194/query_phenotype_to_disease"
        data = {"phenotype_ids": phenotype_ids}
        response = requests.post(url, json=data)
        return [i['disease'] for i in response.json()] if response.status_code == 200 else {"error": response.status_code}
    except Exception as e:
        return {"error": str(e)}

def query_phenotype_to_gene(phenotype_ids: list):
    try:
        url = "http://localhost:8194/query_phenotype_to_gene"
        data = {"phenotype_ids": phenotype_ids}
        response = requests.post(url, json=data)
        return [i['gene'] for i in response.json()] if response.status_code == 200 else {"error": response.status_code}
    except Exception as e:
        return {"error": str(e)}

def query_min_subgraph_for_verifier(node_ids: list):
    try:
        url = "http://localhost:8194/query_min_subgraph"
        data = {"node_ids": node_ids}
        response = requests.post(url, json=data)
        return response.json() if response.status_code == 200 else {"error": response.status_code}
    except Exception as e:
        return {"error": str(e)}


import pandas as pd
import numpy as np
from typing import List, Optional
from collections import Counter, OrderedDict
def merge_and_sort(*lists, unique=True):
    merged = [item for sublist in lists for item in sublist]
    counts = Counter(merged)

    first_indices = {}
    for idx, item in enumerate(merged):
        if item not in first_indices:
            first_indices[item] = idx
            
    sorted_items = sorted(
        merged,
        key=lambda x: (-counts[x], first_indices[x]) 
    )
    
    if unique:
        sorted_items = list(OrderedDict.fromkeys(sorted_items))
        
    return sorted_items
def query_one_hop_gene_disease(
    gene_symbol: str,
    nodes_csv: str = "/share/gguilin/rd-project/KGRD/src/KGRD_framework/kg/merged_nodes_neo4j.csv",
    edges_csv: str = "/share/gguilin/rd-project/KGRD/src/KGRD_framework/kg/merged_edges_neo4j.cleaned.csv",
    top_k = 100
) -> List[str]:


    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)

    nodes = nodes.rename(columns={
        ':ID': 'id', 'name:string': 'name', ':LABEL': 'label', 'source:string': 'source'
    })
    edges = edges.rename(columns={
        ':START_ID': 'start_id', ':END_ID': 'end_id', ':TYPE': 'etype'
    })

    for col in ['id', 'name', 'label']:
        if col not in nodes.columns:
            raise ValueError(f"nodes 缺少必要列: {col}")
    for col in ['start_id', 'end_id', 'etype']:
        if col not in edges.columns:
            raise ValueError(f"edges 缺少必要列: {col}")


    is_gene = nodes['label'].astype(str).str.contains('gene', case=False, na=False)
    is_disease = nodes['label'].astype(str).str.contains('disease', case=False, na=False)

    gene_nodes = nodes.loc[is_gene, ['id', 'name']].copy()
    dis_nodes  = nodes.loc[is_disease, ['id', 'name']].copy()

    gene_nodes['name_up'] = gene_nodes['name'].astype(str).str.upper()


    target_up = str(gene_symbol).upper().strip()
    target_gene_ids = set(gene_nodes.loc[gene_nodes['name_up'] == target_up, 'id'])

    if not target_gene_ids:
        return []


    gene_ids_all = set(gene_nodes['id'])
    dis_ids_all  = set(dis_nodes['id'])

    etype_lower = edges['etype'].astype(str).str.lower()
    if (etype_lower == 'dis_gene').any():
        dg_edges = edges.loc[etype_lower == 'dis_gene', ['start_id', 'end_id']].copy()
    else:
        start_is_gene = edges['start_id'].isin(gene_ids_all)
        end_is_gene   = edges['end_id'].isin(gene_ids_all)
        start_is_dis  = edges['start_id'].isin(dis_ids_all)
        end_is_dis    = edges['end_id'].isin(dis_ids_all)
        mask = (start_is_gene & end_is_dis) | (start_is_dis & end_is_gene)
        dg_edges = edges.loc[mask, ['start_id', 'end_id']].copy()


    dg_edges['gene_id'] = np.where(dg_edges['start_id'].isin(gene_ids_all), dg_edges['start_id'], dg_edges['end_id'])
    dg_edges['disease_id'] = np.where(dg_edges['start_id'].isin(dis_ids_all), dg_edges['start_id'], dg_edges['end_id'])
    dg_edges = dg_edges[['gene_id', 'disease_id']].drop_duplicates()


    id2name = nodes.set_index('id')['name']
    disease_ids = dg_edges.loc[dg_edges['gene_id'].isin(target_gene_ids), 'disease_id']
    diseases = [id2name.get(i) for i in disease_ids if pd.notna(i)]
    diseases = sorted(set([d for d in diseases if isinstance(d, str) and d.strip()]))

    if top_k is not None:
        diseases = diseases[:top_k]
    return diseases


import networkx as nx

def build_nx_graph(data):

    G = nx.Graph()
    for rel in data[0].get('rels', []):
        subject = rel[0].get('entity_id')
        object_ = rel[2].get('entity_id')
        if subject and object_:
            G.add_edge(subject, object_)
    return G

def pg2d_shortest_paths(entitiesdata, entities):
    G = build_nx_graph(entitiesdata)
    start_nodes = entities[:-1]
    end_node = entities[-1]

    for start in start_nodes:
        try:
            paths = list(nx.all_shortest_paths(G, start, end_node))
            return paths
        except nx.NetworkXNoPath:
            print(f"No path from {start} to {end_node}")


from typing import List

from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import AssistantMessage, LLMMessage


class OllamaReasoningModelContext(UnboundedChatCompletionContext):
    """A model context for reasoning models."""

    async def get_messages(self) -> List[LLMMessage]:
        messages = await super().get_messages()
        # Filter out thought field from AssistantMessage.
        messages_out: List[LLMMessage] = []
        for message in messages:
            if isinstance(message, AssistantMessage):
                message.content = message.content.split('</think>')[-1]
            messages_out.append(message)
        return messages_out
    


class ReasoningModelContext(UnboundedChatCompletionContext):
    async def get_messages(self) -> List[LLMMessage]:
        messages = await super().get_messages()
        # Filter out thought field from AssistantMessage.
        messages_out: List[LLMMessage] = []
        for message in messages:
            if isinstance(message, AssistantMessage):
                message.thought = None
            messages_out.append(message)
        return messages_out

class BeforeToolModelContext(UnboundedChatCompletionContext):
    async def get_messages(self) -> List[LLMMessage]:
        messages = await super().get_messages()
        # Filter out thought field from AssistantMessage.
        messages_out: List[LLMMessage] = []
        for message in messages:
            if message.source=='Patient_info':
                messages_out.append(message)
        return messages_out


from typing import List
import json

class afterToolsjsonContext(UnboundedChatCompletionContext):
    async def get_messages(self) -> List[LLMMessage]:
        messages = await super().get_messages()
        messages_out: List[LLMMessage] = []

        for message in messages:
            if isinstance(message, AssistantMessage) and isinstance(message.content, str) and 'json' in message.content:
                # 解析 JSON，失败就给空 dict，避免抛异常
                try:
                    data = parse_json(message.content) or {}
                except Exception:
                    data = {}

                ExpectedOutput = data.get("ExpectedOutput", {}) or {}

                # 统一拿到 TopDiseases 列表
                top = ExpectedOutput.get("TopDiseases", []) or []
                if isinstance(top, (str, dict)):   # 兼容错误类型
                    top = [top]

                # 提取可显示的疾病名；优先常见键名，不存在就转成字符串
                def pick_name(x):
                    if isinstance(x, str):
                        return x.strip()
                    if isinstance(x, dict):
                        for k in ("name", "disease", "Disease", "label", "text", "title", "id"):
                            v = x.get(k)
                            if isinstance(v, (str, int, float)):
                                return str(v)
                        # 全都没有就序列化成紧凑 JSON
                        return json.dumps(x, ensure_ascii=False)
                    return str(x)

                TopDiseases = ",".join(
                    s for s in (pick_name(x) for x in top) if s
                )

                # 处理 OneLineSummary，保证是句子末尾有句点
                one_line = ExpectedOutput.get("OneLineSummary")
                summary = (str(one_line).strip() if one_line is not None else "")
                if summary and not summary.endswith("."):
                    summary += "."

                # 组装新的 message.content（只保留我们需要的内容）
                message.content = (
                    f"{summary} Other matched disease: {TopDiseases}" if TopDiseases else summary
                )

            messages_out.append(message)

        return messages_out


import ast
from typing import List, Union, Any

def PhenoDMiner_tool_call(phenotype_ids: Union[List[str], str]) -> List[Any]:
    """
    Query diseases based on phenotype IDs or names.
    """
    # 1. Input Sanitization: Handle string input safely
    if isinstance(phenotype_ids, str):
        try:
            # Use ast.literal_eval instead of eval to prevent code injection attacks
            parsed = ast.literal_eval(phenotype_ids)
            if isinstance(parsed, list):
                phenotype_ids = parsed
            else:
                # If parsed result is not a list (e.g., just a string), wrap it
                phenotype_ids = [phenotype_ids]
        except (ValueError, SyntaxError):
            # If parsing fails, treat the input string as a single phenotype item
            phenotype_ids = [phenotype_ids]
            
    # 2. Validation: Ensure phenotype_ids is definitely a list
    if not isinstance(phenotype_ids, list):
        phenotype_ids = []

    # 3. Execute Standard Query
    # Added try-except to prevent database errors from crashing the function
    try:
        query_results = query_phenotype_to_disease(phenotype_ids)[:50]
    except Exception as e:
        print(f"Query Error: {e}")
        query_results = []

    # 4. Execute Model Prediction
    prediction_results = []
    try:
        # Note: Instantiating the model here might be slow (see optimization section below)
        pheno_brain_model = PhenoBrain() 
        
        # Convert phenotype names to HPO IDs
        hpo_list = [hpname_to_hpo(i) for i in phenotype_ids]
        
        prediction_results = pheno_brain_model.predict_disease_with_details(hpo_list)[:50]
    except Exception as e:
        # Catch specific errors and log them instead of failing silently
        print(f"Prediction Error: {e}")
        prediction_results = []

    # 5. Merge Results
    return prediction_results + query_results