# KGRD: Knowledge-Graph-Augmented Automated Reasoning Framework for Diagnosis and Counselling of Paediatric Rare Genetic Disorders

This repository contains the **data and source code** associated with the paper:

**KGRD: A Knowledge-Graph-Augmented Automated Reasoning Framework for Diagnosis and Counselling of Paediatric Rare Genetic Disorders**

The manuscript has been **completed**, and the corresponding codebase is **currently under active organization and refinement**.  
Additional components, documentation, and reproducibility scripts will be released progressively.

---

## Repository Structure

```
data/
├── KGRD_diagnosis_test/          # Evaluation datasets for rare disease diagnosis
│   ├── UDN/                    
│   ├── MAC/                      
│   └── Pheval_Rare_Dup/          evaluation data
├── KG/                           # Rare Disease Knowledge Graph 
└── GeneAgent/
    └── RDLinker-att/             # RDLinker-att model data
        └── test/                 # Test set
│
├── docs/
├── quick_start/
├── src/
├── web/
│
├── requirements.txt
└── README.md
```

---

##  Data

##  Model

The RDLinker-att model checkpoints are publicly archived and available for download at [RDLinker-att](https://huggingface.co/Sirius20412/RDLinker-att).


---

## ⚙️ Requirements

All dependencies are listed in:

requirements.txt

You are encouraged to create a virtual environment before installation.

---

## 📜 License

This project is released under the **MIT License**.  
See the `LICENSE` file for details.

---

## 📌 Citation

If you use this code or data in your research, please cite the corresponding paper (citation information will be added upon publication).

---

## 📬 Contact

For questions, issues, or collaboration inquiries, please open an issue in this repository.
