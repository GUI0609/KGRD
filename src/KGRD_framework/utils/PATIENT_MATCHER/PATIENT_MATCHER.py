import os
import sys
import json
import asyncio
import torch
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
with open('PATH/TO/config.json', 'r') as f:
    config = json.load(f)

class SapBertService:
    def __init__(self, config, hpo_to_name_func):
        """
        Initialize SapBERT Service using the centralized config.
        """
        self.config = config
        self.paths = config.get('PATHS', {})
        self.hpo_to_name = hpo_to_name_func
        
        # Use device from config if available (checking RD_LINKER as reference) or default to cuda
        self.device = torch.device(config.get('RD_LINKER', {}).get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Parameters (Defaults if not in JSON)
        self.batch_size = 32
        self.top_k = 100
        self.degree_alpha = 0.75  # Controls weight of common vs rare terms
        
        # Load Model from PATHS['SAPBERT']
        model_path = self.paths.get('SAPBERT')
        if not model_path:
            raise ValueError("SAPBERT path not found in config['PATHS']")
            
        print(f"Loading SapBERT model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

        # Load Degree Info (Assuming it's in the same directory as ENTITY_ID_CSV or specified)
        # If the file is missing in the new config, we default weights to 1.0
        self.hp_name2degree = {}
        try:
            # We look for 'hp_degree.csv' in the same folder as the OBO or entity CSV
            deg_path = os.path.join(os.path.dirname(self.paths.get('ENTITY_ID_CSV', '')), 'hp_degree.csv')
            if os.path.exists(deg_path):
                df_deg = pd.read_csv(deg_path)
                df_deg.columns = [c.strip() for c in df_deg.columns]
                df_deg['hp_name'] = df_deg['hp_name'].astype(str).str.strip()
                df_deg['degree'] = pd.to_numeric(df_deg['degree'], errors='coerce').fillna(0)
                self.hp_name2degree = df_deg.set_index('hp_name')['degree'].to_dict()
                print(f"Loaded degree info for {len(self.hp_name2degree)} HPO terms.")
        except Exception as e:
            print(f"Warning: Could not load degree weights: {e}. Using uniform weighting.")

        # Pre-load Real Patients from PATHS['PATIENT_COHORT']
        self.real_patients = []
        self.real_id_hpo_dict = {}
        patient_path = self.paths.get('PATIENT_COHORT')
        if patient_path and os.path.exists(patient_path):
            with open(patient_path, 'r') as f:
                for line in f:
                    try:
                        p = json.loads(line.strip())
                        self.real_patients.append(p)
                        if p.get('id') and p.get('positive_phenotypes'):
                            self.real_id_hpo_dict[p['id']] = p['positive_phenotypes']
                    except:
                        continue
            print(f"Loaded {len(self.real_id_hpo_dict)} patients from cohort.")

    def _get_inv_degree_weight(self, hp_name):
        d = self.hp_name2degree.get((hp_name or "").strip())
        if d is None or d <= 0: return 1.0
        return 1.0 / ((float(d) + 1e-6) ** self.degree_alpha)

    def match_patients(self, batch_id_hpo_dict):
        # Merge test batch with real cohort for vectorization
        full_dict = {f"test_{k}": v for k, v in batch_id_hpo_dict.items()}
        full_dict.update(self.real_id_hpo_dict)

        # Convert HPO IDs to Names
        id_to_names = {}
        for pid, hpos in full_dict.items():
            names = []
            for h in hpos:
                if isinstance(h, str) and h.startswith("HP:"):
                    names.append(self.hpo_to_name(h))
                else:
                    names.append(str(h))
            id_to_names[pid] = names

        all_sentences = [n for names in id_to_names.values() for n in names]
        owner_ids = [pid for pid, names in id_to_names.items() for _ in names]

        if not all_sentences:
            return []

        # Encoding in batches
        all_embeddings = []
        for i in range(0, len(all_sentences), self.batch_size):
            batch_texts = all_sentences[i:i + self.batch_size]
            with torch.no_grad():
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                outputs = self.model(**inputs)
                # CLS token embedding normalized
                embeds = F.normalize(outputs.last_hidden_state[:, 0, :], p=2, dim=1)
                all_embeddings.append(embeds)
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        term_weights = torch.tensor([self._get_inv_degree_weight(n) for n in all_sentences], 
                                    dtype=torch.float32, device=self.device)

        # Aggregate Patient Vectors (Weighted Average of Phenotypes)
        patient_vectors = {}
        for pid in id_to_names.keys():
            indices = [idx for idx, owner in enumerate(owner_ids) if owner == pid]
            if not indices: continue
            
            embeds = all_embeddings[indices, :]
            weights = term_weights[indices].unsqueeze(1)
            wsum = weights.sum()
            
            vec = (embeds * (weights / (wsum + 1e-9))).sum(dim=0) if wsum > 0 else embeds.mean(dim=0)
            patient_vectors[pid] = F.normalize(vec, p=2, dim=0)

        # Similarity Calculation
        test_ids = [k for k in patient_vectors if k.startswith("test_")]
        real_ids = [k for k in patient_vectors if not k.startswith("test_")]
        
        real_patient_lookup = {p['id']: p for p in self.real_patients}
        results = []

        for test_id in test_ids:
            test_vec = patient_vectors[test_id].unsqueeze(0)
            sims = []
            for rid in real_ids:
                s = F.cosine_similarity(test_vec, patient_vectors[rid].unsqueeze(0)).item()
                sims.append((rid, s))
            
            top_matches = sorted(sims, key=lambda x: x[1], reverse=True)[:self.top_k]
            
            match_details = []
            for rid, score in top_matches:
                pdata = real_patient_lookup.get(rid)
                if pdata:
                    match_details.append({
                        "patient_id": rid,
                        "disease_name": pdata.get('disease_name', 'unknown'),
                        "positive_phenotypes": pdata.get('positive_phenotypes', []),
                        "similarity": round(score, 4)
                    })
            results.append({"query_id": test_id.replace("test_", ""), "top_matches": match_details})

        return results

import json
import os
import sys
from flask import Flask, request, jsonify

# Update System Path for Utils if necessary
utils_path = os.path.dirname(config['PATHS'].get('HPO_OBO', ''))
if utils_path not in sys.path:
    sys.path.append(utils_path)

# Try to import hpo_to_name from your framework's utils
try:
    # Assuming a structure like KGRD_framework.utils
    from utils import hpo_to_name
except ImportError:
    def hpo_to_name(h): 
        return h


app = Flask(__name__)

# Initialize Service
print("Initializing SapBERT Service...")
sapbert_service = SapBertService(config, hpo_to_name)

@app.route("/sapbert_match_patients", methods=["POST"])
def sapbert_match_patients():
    """
    Endpoint for matching patients using SapBERT semantic similarity.
    Expects JSON: {"batch_id_hpo_dict": {"patient1": ["HP:0001", "HP:0002"]}}
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON data provided"}), 400
            
        batch_dict = data.get("batch_id_hpo_dict")
        if not batch_dict:
            return jsonify({"status": "error", "message": "Missing batch_id_hpo_dict"}), 400
        
        results = sapbert_service.match_patients(batch_dict)
        return jsonify({"status": "success", "results": results})
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(error_msg)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "running", "model": "SapBERT"})

if __name__ == "__main__":
    # Pull port from URLS if defined, or default (e.g., matching SHEPHERD at 6006)
    # The JSON shows SHEPHERD: http://localhost:6006
    host = "0.0.0.0"
    port = 6006 
    
    print(f"Starting SapBERT Matching Server on {host}:{port}")
    app.run(host=host, port=port, debug=False)