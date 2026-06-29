# KGRD: Knowledge-Graph-Augmented Automated Reasoning Framework for Diagnosis and Counselling of Paediatric Rare Genetic Disorders

This repository contains the official dataset and source code for the paper:

KGRD: A Knowledge-Graph-Augmented Automated Reasoning Framework for Diagnosis and Counselling of Paediatric Rare Genetic Disorders

## 🚀 Overview

KGRD is a novel framework designed to enhance the diagnosis and genetic counseling of pediatric rare diseases. By integrating automated reasoning agents with comprehensive knowledge graphs, KGRD facilitates more accurate phenotype analysis and pathogenic variant prioritization.

## 🛠️ Installation & Environment Setup

### 1. Repository Cloning

First, clone the repository and navigate to the project root directory:
```
git clone https://github.com/GUI0609/KGRD.git
cd KGRD
```

### 2. Python Environments

To ensure dependency compatibility, this framework requires two distinct Python environments:

1. agent.

2. GCN: For RDLinker.

We recommend using Anaconda/Miniconda to manage these environments:

### Create and activate the Agent environment
```
conda create -n agent python=3.12.2
pip install -r requirements.txt
```
### Create and activate the GCN environment
```
conda create -n GCN python=3.8.20
pip install -r requirements_RDLinker.txt
```


## 📂 Data Preparation & Dependencies

Before running the framework, you must download several external components and model checkpoints. Ensure they are placed in the specific directories outlined below.

### 1. External Components & Models
Please download and move the following resources to their respective destinations:
https://zenodo.org/records/18174736
* **RDLinker-att Model Checkpoints**
    * **Source:** [HuggingFace - RDLinker-att](https://huggingface.co/Sirius20412/RDLinker-att)
    * **Destination:** `src/KGRD_framework/utils/RDLinker-att`
* **Knowledge Graph (KG) & Datasets**
    * **Location:** `src/KGRD_framework/kg`
    * **Data Source:** [Zenodo - KGRD Knowledge Graph](https://zenodo.org/records/18174736)
    * **Use RDLinker-att:** `src/KGRD_framework/utils/RDLinker-att`
        change the PATH/TO/kg AS YOUR kg PATH
    * **Format:** Follows the [TxGNN](https://github.com/mims-harvard/TxGNN) data schema.
* **Doc2Hpo 2.0**
    * **Source:** [Doc2Hpo2.0 GitHub](https://github.com/storm-stout/Doc2Hpo)
    * **Destination:** `src/KGRD_framework/utils/Doc2Hpo2.0`

### 2. Directory Structure Check
Your project structure should look like this after preparation:
```text
KGRD/
├── data/
│   └── KGRD_diagnosis_test/
├── src/
│   └── KGRD_framework/
│       ├── kg/                
│       └── utils/
│           ├── RDLinker-att/  <-- [Place Checkpoints and TxGNN here]
│           └── Doc2Hpo2.0/    <-- [Clone Doc2Hpo2.0 here]

```

### 3. Patient Cohort Data

Prepare your rare disease patient dataset. A reference format is provided in PATHENT_COHORT.txt.

### 4. VCF Preprocessing:
If your raw data is in VCF format, it requires preprocessing to generate valid test samples.

Execute the preprocessing pipeline:
```
bash src/VCF_preprocess/run_rd_pipelines.sh
```

The output will resemble the format found in:
data/KGRD_diagnosis_test/single_test_case.json

In the released JSON files, the historical field name `true_gene` denotes the candidate gene input consumed by GeneAgent.

### 5. Component Verification

To verify that all tools (including the Tool Agent and Verifier Agent) are correctly installed, please run the unit tests provided in: src/KGRD_framework/test_utils.ipynb

## ⚙️ Configuration

Initialize Config File: config.json
Duplicate the example configuration file and rename it.
```
cp src/KGRD_framework/config_example.json src/KGRD_framework/config.json
```

Customize Parameters:
Open `src/KGRD_framework/config.json` and update the following:

Paths: Replace placeholder paths such as `PATH/TO/KGRD` with the absolute paths on your local machine.

LLM Settings: Configure `API_KEYS` and `LLM_MODELS`. The runtime reads `src/KGRD_framework/config.json` by default. To use a different file, set `KGRD_CONFIG_PATH` or pass `--config_path`.

Literature evidence retrieval in the Verifier can use either an authorized private Dify workflow or the public PubMed API. The internal vector-indexed literature corpus used during development is not redistributed because of copyright and licensing restrictions; set `LITERATURE_RETRIEVAL.PROVIDER` to `pubmed` for the public fallback or to `dify` only when you have access to the authorized private workflow. The PubMed backend is a public substitute for reproducible execution and may not exactly reproduce the private vector-indexed literature channel used in the paper.

## 🏃 Usage

1. Start Background Services

Initiate the necessary backend services by running the startup script. Monitor the output (.out) files to ensure all services launch successfully.
```
bash src/KGRD_framework/start_all.sh
```

2. Run Single Case Inference

To run single-case inference, use `main_models.py`. The example below shows a representative DeepSeek-Chat configuration with explicitly selected tool agents.
```
python src/KGRD_framework/main_models.py \
    --config_path src/KGRD_framework/config.json \
    --model_name deepseek \
    --dataset_name single_test_case \
    --project_name single_test_case \
    --stage follow_up \
    --times 1 \
    --num_doctors <num_doctors> \
    --n_round 15 \
    --withtool \
    --SelectTool "PhenoDMiner,GeneDPredictor,PatientDMatcher,KnowledgeVerifier"
```

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for full text and details.

## 📌 Citation
If you find this code or dataset useful for your research, please cite our paper:

(Citation information will be updated upon publication)

## 📬 Contact
For technical questions, bug reports, or collaboration inquiries, please open an issue in this repository.
