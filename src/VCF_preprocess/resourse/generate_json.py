import json
import os
import sys
import yaml

def load_hpo_ontology(file_path):
    """
    Parses the hp.obo file and returns a mapping of HPO identifiers to phenotype labels.
    """
    hpo_mapping = {}
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ontology file not found at: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        current_id = None
        for line in f:
            line = line.strip()
            if line.startswith('id:'):
                current_id = line.split('id:')[1].strip()
            elif line.startswith('name:') and current_id:
                name = line.split('name:')[1].strip()
                hpo_mapping[current_id] = name
                current_id = None
    return hpo_mapping

def get_phenotype_label(hpo_id, hpo_mapping):
    """
    Retrieves the descriptive label for a given HPO identifier.
    """
    return hpo_mapping.get(hpo_id, "Unknown Phenotype")

def parse_hpo_identifiers(file_path):
    """
    Extracts HPO identifiers from a specialized text file.
    """
    hpo_list = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                identifier = line.strip()
                if identifier:
                    hpo_list.append(identifier)
    return hpo_list

def extract_sex_metadata(yaml_path):
    """
    Extracts biological sex information from the source YAML manifest.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Source YAML manifest not found: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    subject = data.get('subject', {})
    sex = subject.get('sex')
    
    if not sex:
        print(f"Warning: Sex attribute missing in {os.path.basename(yaml_path)}. Defaulting to FEMALE.")
        return "FEMALE"
        
    return str(sex).upper()

def serialize_phenopacket(sample_id, sex, base_working_dir, output_directory, hpo_mapping):
    """
    Constructs and serializes a Phenopacket JSON object adhering to version 2.0.0 schema.
    """
    # Construct input paths
    hpo_input_path = os.path.join(base_working_dir, 'hpos', f"{sample_id}.hpos.txt")
    vcf_resource_path = os.path.join(base_working_dir, 'vcf', f"{sample_id}.vcf")

    if not os.path.exists(hpo_input_path):
        print(f"Error: Required HPO list missing: {hpo_input_path}")
        return

    # Retrieve phenotypic features
    hpo_ids = parse_hpo_identifiers(hpo_input_path)
    phenotypic_features = []
    for hpo_id in hpo_ids:
        feature = {
            "type": {
                "id": hpo_id,
                "label": get_phenotype_label(hpo_id, hpo_mapping)
            }
        }
        phenotypic_features.append(feature)

    # =========================================================================
    # Phenopacket Schema v2.0.0 Structure (Immutable)
    # =========================================================================
    phenopacket = {
        "id": sample_id,
        "subject": {
            "id": sample_id,
            "sex": sex.upper()
        },
        "phenotypicFeatures": phenotypic_features,
        "files": [{
            "uri": f"file://{vcf_resource_path}",
            "fileAttributes": {
                "fileFormat": "vcf",
                "genomeAssembly": "GRCh37"
            }
        }],
        "metaData": {
            "phenopacketSchemaVersion": "2.0.0",
            "resources": [
                {
                    "id": "hp",
                    "name": "human phenotype ontology",
                    "url": "http://purl.obolibrary.org/obo/hp.owl",
                    "version": "2019-10-16",
                    "namespacePrefix": "HP",
                    "iriPrefix": "http://purl.obolibrary.org/obo/HP_"
                },
                {
                    "id": "eco",
                    "name": "Evidence and Conclusion Ontology",
                    "url": "http://purl.obolibrary.org/obo/eco.owl",
                    "version": "2019-10-16",
                    "namespacePrefix": "ECO",
                    "iriPrefix": "http://purl.obolibrary.org/obo/ECO_"
                }
            ]
        }
    }
    # =========================================================================

    os.makedirs(output_directory, exist_ok=True)
    
    output_file_path = os.path.join(output_directory, f"{sample_id}.json")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(phenopacket, f, indent=2)

    print(f"Success: Phenopacket JSON generated at {output_file_path}")


if __name__ == "__main__":
    # 1. Argument Validation
    if len(sys.argv) < 2:
        print("Usage: python generate_json.py <SAMPLE_ID>")
        sys.exit(1)

    input_sample_id = sys.argv[1]

    # ================= USER CONFIGURATION AREA =================
    # Define relative or absolute paths for the execution environment.
    
    # Directory containing source YAML files for sex metadata extraction
    RAW_PHENOTYPE_DIR = "./raw_phenotype"
    
    # Base directory containing 'hpos/' and 'vcf/' subdirectories
    BASE_WORKING_DIR = "./"
    
    # Target directory for generated Phenopacket JSON files
    OUTPUT_DIR = "./phenopackets"
    
    # Path to the Human Phenotype Ontology (HPO) OBO file
    HPO_OBO_FILE_PATH = "../../SHEPHERD/patients/und_data/hpo/2019/hp.obo"
    # ===========================================================

    # Resolve paths relative to the script's location
    base_script_path = os.path.dirname(os.path.abspath(__file__))
    
    def resolve_path(p):
        if os.path.isabs(p): return p
        return os.path.normpath(os.path.join(base_script_path, p))

    abs_raw_dir = resolve_path(RAW_PHENOTYPE_DIR)
    abs_base_dir = resolve_path(BASE_WORKING_DIR)
    abs_out_dir = resolve_path(OUTPUT_DIR)
    abs_hpo_path = resolve_path(HPO_OBO_FILE_PATH)

    # 2. Extract Metadata
    yaml_source = os.path.join(abs_raw_dir, f"{input_sample_id}.yml")
    try:
        extracted_sex = extract_sex_metadata(yaml_source)
        print(f"Status: Metadata extraction successful. Sex: {extracted_sex}")
    except FileNotFoundError:
        print(f"Critical Error: YAML manifest missing for sample {input_sample_id} at {yaml_source}")
        sys.exit(1)
    except Exception as e:
        print(f"Critical Error: Metadata extraction failed: {e}")
        sys.exit(1)

    # 3. Initialize Ontology Dictionary
    try:
        ontology_mapping = load_hpo_ontology(abs_hpo_path)
    except Exception as e:
        print(f"Critical Error: Failed to initialize HPO ontology: {e}")
        sys.exit(1)

    # 4. Execute Serialization
    serialize_phenopacket(input_sample_id, extracted_sex, abs_base_dir, abs_out_dir, ontology_mapping)