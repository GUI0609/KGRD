import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any

class MedicalEvaluator:
    def __init__(self, config_path: str = "PATH/TO/config.json"):
        self.config = self._load_config(config_path)
        
        self.provider = self.config.get("LLM_PROVIDER", "deepseek").lower()
        
        api_keys = self.config.get("API_KEYS", {})
        self.api_key = api_keys.get(self.provider.upper(), "")
        
        llm_settings = self.config.get("LLM_MODELS", {}).get(self.provider, {})
        self.model_name = llm_settings.get("model", "Unknown_Model")
        
        paths = self.config.get("PATHS", {})
        self.output_dir = Path(paths.get("OUTPUT_ROOT", "./output"))
        self.eval_dir = Path(paths.get("EVAL_DIR", "./eval")) 
        self.dataset_path = Path(paths.get("PATIENT_COHORT", "")) if paths.get("PATIENT_COHORT") else None

        self._setup_logging()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, path: str) -> Dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _setup_logging(self):
        evaluator_settings = self.config.get("LLM_MODELS", {}).get("evaluator", {})
        log_level = evaluator_settings.get("log_level", "INFO")
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _safe_str(x: Any) -> str:
        return x.strip() if isinstance(x, str) else ''

    def _load_reference_dataset(self) -> Dict[str, Any]:
        if not self.dataset_path or not self.dataset_path.exists():
            self.logger.warning(f"Reference dataset path invalid or missing: {self.dataset_path}")
            return {}
        
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                if self.dataset_path.suffix == '.json':
                    data = json.load(f)
                    return {str(item.get("Case URL", "")): item for item in data.get("Cases", [])}
                else:
                    self.logger.info("Dataset is not in JSON format, skipping structured cross-reference")
                    return {}
        except Exception as e:
            self.logger.error(f"Failed to load reference dataset: {e}")
            return {}

    def run_evaluation(self) -> Tuple[pd.DataFrame, Dict]:
        dataset_cases = self._load_reference_dataset()
        
        ml_scores_counter = Counter()
        op_scores_counter = Counter()
        ml_values = []
        op_values = []
        lower_op_files = []
        records = []

        self.logger.info(f"Starting evaluation for model: {self.model_name} (Provider: {self.provider})")

        if not self.eval_dir.exists():
            self.logger.error(f"Evaluation directory does not exist: {self.eval_dir}")
            return pd.DataFrame(), {}

        for file_path in self.eval_dir.rglob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                filename = data.get("Filename", file_path.stem)
                
                ml_eval = data.get("Most Likely Evaluation") or data.get("Most Likely") or data.get("Name")
                ml_score = None
                if isinstance(ml_eval, dict) and "Score" in ml_eval:
                    try:
                        ml_score = float(ml_eval["Score"])
                        ml_scores_counter[str(ml_score)] += 1
                        ml_values.append(ml_score)
                    except (ValueError, TypeError):
                        pass

                op_eval = data.get("Other Possible Evaluation") or data.get("Other Possible") or data.get("Differential Diagnosis")
                op_score = None
                if isinstance(op_eval, dict) and "Score" in op_eval:
                    try:
                        op_score = float(op_eval["Score"])
                    except (ValueError, TypeError):
                        pass

                if ml_score is not None and op_score is not None:
                    if op_score < ml_score:
                        lower_op_files.append((str(file_path), ml_score, op_score))
                        op_score = ml_score 
                    
                    op_scores_counter[str(op_score)] += 1
                    op_values.append(op_score)

                ml_diag = self._safe_str(data.get("Most Likely Diagnosis") or data.get("Most Likely") or "")
                diff_diag = data.get("Differential Diagnosis") or data.get("Other Possible")

                case_info = dataset_cases.get(filename, {})
                if not case_info:
                    case_info = dataset_cases.get(filename.replace(".json", ""), {})
                
                records.append({
                    "Filename": filename,
                    "Final Name": case_info.get("Final Name", ""),
                    "Case URL": case_info.get("Case URL", ""),
                    "Most Likely Diagnosis": ml_diag,
                    "Differential Diagnosis": diff_diag,
                    "Most Likely Score": ml_score,
                    "Other Possible Score": op_score,
                    "Raw_ML_Eval": ml_eval if isinstance(ml_eval, dict) else None,
                    "Raw_CD_Eval": op_eval if isinstance(op_eval, dict) else None
                })

            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}")

        results_df = pd.DataFrame(records)
        
        summary = {
            "model_name": self.model_name,
            "provider": self.provider,
            "most_likely": {
                "average": float(np.mean(ml_values)) if ml_values else 0.0,
                "std_dev": float(np.std(ml_values)) if ml_values else 0.0,
                "count": len(ml_values),
                "distribution": dict(ml_scores_counter)
            },
            "other_possible": {
                "average": float(np.mean(op_values)) if op_values else 0.0,
                "std_dev": float(np.std(op_values)) if op_values else 0.0,
                "count": len(op_values),
                "distribution": dict(op_scores_counter)
            },
            "mismatch_corrections_count": len(lower_op_files)
        }

        self._print_results(summary)
        self._save_summary(summary)

        return results_df, summary

    def _print_results(self, summary: Dict):
        print(f"\n{'='*50}")
        print(f" EVALUATION REPORT: {summary['model_name']} ({summary['provider'].upper()})")
        print(f"{'='*50}")
        
        ml = summary['most_likely']
        print(f"\n[Most Likely Evaluation]")
        print(f"  Count:   {ml['count']}")
        print(f"  Average: {ml['average']:.2f}")
        print(f"  Std Dev: {ml['std_dev']:.2f}")
        
        op = summary['other_possible']
        print(f"\n[Other Possible Evaluation (Corrected)]")
        print(f"  Count:   {op['count']}")
        print(f"  Average: {op['average']:.2f}")
        print(f"  Std Dev: {op['std_dev']:.2f}")
        
        print(f"\n[Alerts]")
        print(f"  Files with CD < ML (Auto-corrected): {summary['mismatch_corrections_count']}")
        print(f"{'='*50}\n")

    def _save_summary(self, summary: Dict):
        save_enabled = self.config.get("LLM_MODELS", {}).get("evaluator", {}).get("save_summary", True)
        if not save_enabled:
            return
            
        output_path = self.output_dir / f"{self.model_name}_summary.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Summary saved to {output_path}")

if __name__ == "__main__":
    CONFIG_PATH = 'PATH/TO/config.json'
    
    try:
        evaluator = MedicalEvaluator(CONFIG_PATH)
        df, stats = evaluator.run_evaluation()
        
        if evaluator.api_key:
            print(f"API Key for {evaluator.provider} loaded (length: {len(evaluator.api_key)})")
            
    except Exception as e:
        print(f"Application failed: {e}")