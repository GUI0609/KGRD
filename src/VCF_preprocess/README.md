# Rare Disease Prioritization Pipeline

## Overview

This repository implements a rare disease genomic analysis pipeline that integrates variant prioritization and phenotype-driven gene ranking. The pipeline accepts a VCF file and a phenotype file in YAML format, executes a standardized workflow, and outputs ranked candidate gene lists according to three methods: **LIRICAL**, **Exomiser**, and **AI-MARRVEL**.

## Directory Structure

```text
.
├── run_rd_pipelines.sh          # Main execution script
├── testdata_dir/
│   ├── vcf/                     # Input directory for VCF files
│   └── raw_phenotype/           # Input directory for Phenotype YAML files
├── output_dir/
│   └── list_output/             # Output directory for ranked lists
└── resourse/                    # Helper scripts and resources
    └── batch_run_scheduler.py   # Script for batch processing
```

## Prerequisites

Ensure the following software tools are downloaded and installed prior to workflow execution:

- **LIRICAL**: A phenotype-driven prioritization tool.  
  GitHub: https://github.com/TheJacksonLaboratory/LIRICAL  
  Documentation: https://lirical.readthedocs.io/

- **Exomiser**: A variant annotation and prioritization tool.  
  GitHub: https://github.com/exomiser/Exomiser  
  Data & releases: https://data.monarchinitiative.org/exomiser/latest

- **AI-MARRVEL (AIM)**: An AI-driven prioritization system for rare disease diagnostics.  
  GitHub: https://github.com/LiuzLab/AI_MARRVEL

Ensure all executables are accessible in your environment (e.g., added to `PATH`).

## Data Preparation

1. Upload the sample VCF file to:

```
testdata_dir/vcf/sample.vcf
```

2. Upload the sample phenotype YAML file to:

```
testdata_dir/raw_phenotype/sample.yml
```

## Pipeline Execution

Run the pipeline using:

```
./run_rd_pipelines.sh <sample_identifier>
```

Example:

```
./run_rd_pipelines.sh BC241523OO
```

## Output

Results are written to:

```
output_dir/list_output
```

Generated files include:

- `aim_list/<sample>`
- `exomiser_list/<sample>`
- `lirical_list_/<sample>`

Each file represents ranked candidate genes from the corresponding method.

## Batch Processing

For batch execution across multiple samples, use:

```
resourse/batch_run_scheduler.py
```

This script schedules and executes pipeline runs for multiple sample identifiers.

