KGRD: Knowledge-Graph-Augmented Automated Reasoning Framework for Diagnosis and Counselling of Paediatric Rare Genetic Disorders

This repository contains the official dataset and source code for the paper:

KGRD: A Knowledge-Graph-Augmented Automated Reasoning Framework for Diagnosis and Counselling of Paediatric Rare Genetic Disorders

🚀 Overview

KGRD is a novel framework designed to enhance the diagnosis and genetic counseling of pediatric rare diseases. By integrating automated reasoning agents with comprehensive knowledge graphs, KGRD facilitates more accurate phenotype analysis and pathogenic variant prioritization.

🛠️ Installation & Environment Setup

1. Repository Cloning

First, clone the repository and navigate to the project root directory:

git clone [https://github.com/your-username/KGRD.git](https://github.com/your-username/KGRD.git)
cd KGRD


2. Python Environments

To ensure dependency compatibility, this framework requires two distinct Python environments:

agent: For the reasoning agents and LLM interactions.

GCN: For the Graph Convolutional Network modules (RDLinker).

We recommend using Anaconda/Miniconda to manage these environments:

# Create and activate the Agent environment
conda create -n agent python=3.12.2
pip install -r requirements.txt

# Create and activate the GCN environment
conda create -n GCN python=3.8.20
pip install -r requirements_RDLinker.txt


📂 Data Preparation & Dependencies

Before running the framework, several external components and model checkpoints must be downloaded and placed in specific directories.

1. RDLinker-att Model Checkpoints

The pre-trained checkpoints for the RDLinker-att model are publicly available.

Download Source: HuggingFace - RDLinker-att

Destination: Move the downloaded files to:
src/KGRD_framework/utils/RDLinker-att

2. Knowledge Graph & Training Data

The core Knowledge Graph (KG) and training/testing datasets should be located in src/KGRD_framework/kg.

Data Format: For details regarding the data schema and formatting, please refer to the TxGNN repository.

3. Doc2Hpo 2.0

The framework utilizes Doc2Hpo for phenotype extraction.

Download Source: Doc2Hpo2.0 GitHub

Destination: Place the repository in:
src/KGRD_framework/utils/Doc2Hpo2.0

4. Patient Cohort Data

Prepare your rare disease patient dataset. A reference format is provided in PATHENT_COHORT.txt.

VCF Preprocessing:
If your raw data is in VCF format, it requires preprocessing to generate valid test samples.

Execute the preprocessing pipeline:

bash src/VCF_preprocess/run_rd_pipelines.sh


The output will resemble the format found in:
data/KGRD_diagnosis_test/single_test_case.json

5. Component Verification

To verify that all tools (including the Tool Agent and Verifier Agent) are correctly installed, please run the unit tests provided in:
src/KGRD_framework/test_utils.ipynb

⚙️ Configuration

Initialize Config File:
Duplicate the example configuration file and rename it.

cp config_example.json config.json


Customize Parameters:
Open config.json and update the following:

Paths: Replace all instances of PATH/TO/config.json (and other placeholder paths) with the absolute paths on your local machine.

LLM Settings: Configure the API keys and parameters for the Large Language Model.

🏃 Usage

1. Start Background Services

Initiate the necessary backend services by running the startup script. Monitor the output (.out) files to ensure all services launch successfully.

bash src/KGRD_framework/start_all.sh


2. Run Single Case Inference

To test the framework on a single patient case, use the main_models.py script. The command below demonstrates a standard execution using the DeepSeek-Chat model with specific tool selection.

python main_models.py \
    --model_name deepseek-chat \
    --dataset_name single_test_case \
    --project_name single_test_case \
    --stage follow_up \
    --times 1 \
    --num_doctors 9 \
    --n_round 15 \
    --withtool True \
    --SelectTool "PhenoDMiner,GeneDPredictor,PatientDMatcher"




# 📜 License
This project is licensed under the MIT License. See the LICENSE file for full text and details.

# 📌 Citation
If you find this code or dataset useful for your research, please cite our paper:

(Citation information will be updated upon publication)

# 📬 Contact
For technical questions, bug reports, or collaboration inquiries, please open an issue in this repository.