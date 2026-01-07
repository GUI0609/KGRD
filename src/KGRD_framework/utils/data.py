import json
import os.path as osp
from collections import Counter, OrderedDict

import pandas as pd
import numpy as np
from typing import List, Optional
import sys
sys.path.append(osp.join(osp.dirname(__file__), "utils"))
from .utils import query_one_hop_gene_disease,merge_and_sort

config = json.load(open("PATH/TO/config.json","r"))

class MedDataset:
    
    def __init__(self, dataname: str="rare_disease_cases_302"):
        dataname = f"{dataname}.json"
        self.data_path = osp.join(config["PATHS"]["DATASETS"], dataname)
        self.cases = None
        self.url2idx = {}  

        self.load()


    def __len__(self):
        return len(self.cases)

    def load(self):
        with open(self.data_path, "r") as file:
            data = json.load(file)
            self.cases = data["Cases"]

        self.url2idx = {}
        for i, case in enumerate(self.cases):
            case_url = case.get("Case URL")
            if case_url:  
                self.url2idx[case_url] = i

    def __getitem__(self, idx: int):
        case = self.cases[idx]
        
        case_type = case.get("Type")
        case_name = case.get("Final Name")
        case_crl = case.get("Case URL")
        case_initial_presentation = case.get("Initial Presentation")
        case_follow_up_presentation = case.get("Follow-up Presentation")

        p = case.get("p",[])
        p_name = case.get("p_name",[])
        disease_g2d = case.get("disease_g2d",[])[:100]
        disease_p2d = case.get("disease_p2d",[])[:50]
        d_patient_sapbert = case.get("d_patient_sapbert",[])[:100]
        
        d_phenobrain = case.get("d_phenobrain",[])[:50]
        true_gene = case.get("true_gene",[])
        if isinstance(true_gene,str):
            true_gene = [true_gene]

        HPO_LIST = p
        HPO_NAME_LIST = p_name
        GENE = true_gene
        DISEASE_FROM_PHENOTYPE = merge_and_sort(d_phenobrain,disease_p2d)
        DISEASE_FROM_PATIENT_LIKE_ME = d_patient_sapbert


        DISEASE_FROM_GENE = []
        if len(true_gene)>0:
            for i in true_gene:
                DISEASE_FROM_GENE+=query_one_hop_gene_disease(i)
        DISEASE_FROM_GENE+=disease_g2d




        return (case_type, case_name, case_crl,
                case_initial_presentation, case_follow_up_presentation,
                HPO_LIST, HPO_NAME_LIST, GENE, DISEASE_FROM_GENE,
                DISEASE_FROM_PHENOTYPE, DISEASE_FROM_PATIENT_LIKE_ME)

    def get_by_case_url(self, case_url: str):

        idx = self.url2idx.get(case_url)
        if idx is None:
            raise KeyError(f"Case URL {case_url} 不存在")
        return self[idx]


