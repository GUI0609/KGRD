import os
import sys

import json
with open('PATH/TO/config.json', 'r') as f:
    config = json.load(f)

sys.path.append(config['RD_LINKER']['MAIN_PATH'])
from flask import Flask, request, jsonify
import pandas as pd

import torch
import dgl
import tqdm

import math
import copy
import pickle
import numpy as np
import warnings
from tqdm.auto import tqdm



from txgnn.utils import (
    preprocess_kg,
    create_split,
    process_gene_area_split,
    create_dgl_graph,
    evaluate_graph_construct,
    convert2str,
    data_download_wrapper,
)

from txgnn import TxData, TxGNN, TxEval
from txgnn.utils import evaluate_fb

warnings.filterwarnings("ignore")

app = Flask(__name__)

# 全局变量
DATA_PATH = config['RD_LINKER']['DATA_PATH']
MODEL_PATH = config['RD_LINKER']['MODEL_PATH']
SEED = config['RD_LINKER']['SEED']
# 全局变量存储
txgnn = None
best_model = None
device = config['RD_LINKER']['DEVICE']
nodes_data = None
df_ = None
df = None
idx2id_disease = None
id2name_disease = None
disease_nodes = None
G = None

def initialize_model():
    global txgnn, best_model, device, nodes_data, df_, df, idx2id_disease, id2name_disease, disease_nodes, G
    
    # 预加载模型和数据
    nodes_data = pd.read_csv(os.path.join(DATA_PATH, "nodes.csv"))
    df_ = pd.read_csv(os.path.join(DATA_PATH, "kg.csv"))

    # 初始化 txdata 和 txgnn 模型
    txdata = TxData(data_folder_path=DATA_PATH)
    txdata.prepare_split(split="complex_disease", seed=SEED, no_kg=False)

    txgnn = TxGNN(
        data=txdata, weight_bias_track=False, proj_name="TxGNN", exp_name="TxGNN", device=device
    )
    
    # 
    txgnn.load_pretrained(path=MODEL_PATH)
    best_model = txgnn.best_model.to(device)
    G = txgnn.G.to(device)
    
    # 
    txeval = TxEval(model=txgnn)
    df_train = txeval.df_train
    df_valid = txeval.df_valid
    df_test = txeval.df_test
    df = txeval.df
    
    # 
    df["x_id"] = df.x_id.apply(lambda x: convert2str(x))
    df["y_id"] = df.y_id.apply(lambda x: convert2str(x))
    df_["x_id"] = df_.x_id.apply(lambda x: convert2str(x))
    df_["y_id"] = df_.y_id.apply(lambda x: convert2str(x))
    
    # 
    d_node_types = ['disease', 'gene/protein']
    d_rel_types = ['dis_gene', 'clinical feature']
    idx2id_disease = get_idx2id(df, "disease")
    idx2id_gene = get_idx2id(df, "gene/protein")
    idx2id_phenotype = get_idx2id(df, "effect/phenotype")
    
    id2name_disease = get_id2name(df_, "disease")
    id2name_gene = get_id2name(df_, "gene/protein")
    id2name_phenotype = get_id2name(df_, "effect/phenotype")
    
    disease_ids_rels = {}
    gene_ids_rels = {}
    phenotype_ids_rels = {}
    for i in d_rel_types:
        if i == "dis_gene":
            disease_ids_rels["rev_" + i] = df[df.relation == i].x_id.unique()
            gene_ids_rels[i] = df[df.relation == i].y_id.unique()
        elif i == "clinical feature":
            disease_ids_rels["rev_" + i] = df[df.relation == i].x_id.unique()
            phenotype_ids_rels[i] = df[df.relation == i].y_id.unique()
    
    num_of_disease_rels = {}
    num_of_gene_rels = {}
    num_of_phenotype_rels = {}
    for i in d_rel_types:
        if i == "dis_gene":
            num_of_disease_rels["rev_" + i] = len(disease_ids_rels["rev_" + i])
            num_of_gene_rels[i] = len(gene_ids_rels[i])
        elif i == "clinical feature":
            num_of_disease_rels["rev_" + i] = len(disease_ids_rels["rev_" + i])
            num_of_phenotype_rels[i] = len(phenotype_ids_rels[i])
            
    rel = "rev_dis_gene"
    df_train_valid = pd.concat([df_train, df_valid])
    df_train_valid.loc[df_train_valid['relation'] == 'dis_gene', 'relation'] = 'rev_dis_gene'
    df_train_valid.loc[df_train_valid['relation'] == 'rev_dis_gene', ['x_id', 'y_id']] = \
        df_train_valid.loc[df_train_valid['relation'] == 'rev_dis_gene', ['y_id', 'x_id']].values
    df_train_valid.loc[df_train_valid['relation'] == 'rev_dis_gene', ['x_idx', 'y_idx']] = \
        df_train_valid.loc[df_train_valid['relation'] == 'rev_dis_gene', ['y_idx', 'x_idx']].values
    
    df_test.loc[df_test['relation'] == 'dis_gene', 'relation'] = 'rev_dis_gene'
    df_test.loc[df_test['relation'] == 'rev_dis_gene', ['x_id', 'y_id']] = \
        df_test.loc[df_test['relation'] == 'rev_dis_gene', ['y_id', 'x_id']].values
    df_test.loc[df_test['relation'] == 'rev_dis_gene', ['x_idx', 'y_idx']] = \
        df_test.loc[df_test['relation'] == 'rev_dis_gene', ['y_idx', 'x_idx']].values
    
    df_dd = df_test[df_test.relation.isin(d_rel_types)]
    df_dd_train = df_train_valid[df_train_valid.relation.isin(d_rel_types)]
    df_rel_dd = df_dd[df_dd.relation == rel]
    df_rel_dd_train = df_dd_train[df_dd_train.relation == rel]
    gene_nodes = G.nodes("gene/protein").cpu().numpy()
    disease_nodes = G.nodes("disease").cpu().numpy()
    
    print("finish initialize")

def get_idx2id(df, node_type):  
    idx2id_node_type = dict(
        df[df.x_type == node_type][["x_idx", "x_id"]].drop_duplicates().values
    )
    idx2id_node_type.update(
        dict(
            df[df.y_type == node_type][["y_idx", "y_id"]].drop_duplicates().values
        )
    )
    return idx2id_node_type

def get_id2name(df, node_type):
    id2name_node_type = dict(
        df[df.x_type == node_type][["x_id", "x_name"]].drop_duplicates().values
    )
    id2name_node_type.update(
        dict(
            df[df.y_type == node_type][["y_id", "y_name"]].drop_duplicates().values
        )
    )
    return id2name_node_type

def gene_name_to_node_index(gene_name):
    row = nodes_data[(nodes_data.node_type == 'gene/protein') & (nodes_data.node_name == gene_name)]
    if len(row) == 0:
        raise ValueError(f"Gene name '{gene_name}' not found.")
    return int(row.node_index.values[0])

def predict_gene_diseases_by_name(gene_name, topk=100):
    try:
        test_node_id = gene_name_to_node_index(gene_name)
        disease_ids = list(disease_nodes)
        
        src = torch.tensor([test_node_id] * len(disease_ids), dtype=torch.int64, device=device)
        dst = torch.tensor(disease_ids, dtype=torch.int64, device=device)
        
        df_pred = pd.DataFrame({
            'x_idx': src.cpu(),
            'relation': ['rev_dis_gene'] * len(disease_ids),
            'y_idx': dst.cpu()
        })

        pred = txgnn.predict(df_pred)
        
        rel = "rev_dis_gene"
        pred = (pred[("gene/protein", rel, "disease")]
                .reshape(-1)
                .detach()
                .cpu()
                .numpy())

        predictions = []
        for idx, disease_id in enumerate(disease_ids):
            disease_name = id2name_disease.get(idx2id_disease[disease_id], 'unknown')
            predictions.append({
                'disease_name': disease_name,
                'disease_id': idx2id_disease[disease_id],
                'score': float(pred[idx])
            })
        
        predictions.sort(key=lambda x: x['score'], reverse=True)
        return predictions[:topk]
        
    except Exception as e:
        return {'error': str(e)}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'genes' not in data:
            return jsonify({'error': 'no gene list'}), 400
            
        genes = data['genes']
        if not isinstance(genes, list):
            return jsonify({'error': 'genes mush be lish'}), 400
            
        results = {}
        for gene in genes:
            predictions = predict_gene_diseases_by_name(gene)
            results[gene] = predictions
            
        return jsonify({
            'status': 'success',
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'message': '模型已就绪'})

if __name__ == '__main__':

    initialize_model()

    app.run(host='0.0.0.0', port=8092, debug=True) 