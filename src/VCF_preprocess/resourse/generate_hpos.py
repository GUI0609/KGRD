import os
import yaml
import sys

def extract_hpo_terms(input_dir, output_dir):
    """
    Extracts HPO (Human Phenotype Ontology) identifiers from YAML files
     and saves them into individual text files.
    """
    
    # Resolve absolute paths to ensure script robustness
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    def resolve_path(relative_path):
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.normpath(os.path.join(base_dir, relative_path))

    abs_input_path = resolve_path(input_dir)
    abs_output_path = resolve_path(output_dir)

    # 1. Validate input directory existence
    if not os.path.exists(abs_input_path):
        print(f"Error: Input directory does not exist: {abs_input_path}")
        return

    # 2. Verify or initialize output directory
    if not os.path.exists(abs_output_path):
        try:
            os.makedirs(abs_output_path)
            print(f"Status: Creating output directory at {abs_output_path}")
        except OSError as e:
            print(f"Error: Failed to create output directory: {e}")
            return

    print(f"Execution initialized...")
    print(f"Source Directory:      {abs_input_path}")
    print(f"Destination Directory: {abs_output_path}")
    print("-" * 60)

    processed_count = 0
    
    # 3. Iterate through directory contents
    for filename in os.listdir(abs_input_path):
        if filename.lower().endswith((".yml", ".yaml")):
            file_path = os.path.join(abs_input_path, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Parse YAML structure safely
                    data = yaml.safe_load(f)
                
                hpo_ids = []
                
                # 4. Phenotypic feature extraction logic
                if 'phenotypicFeatures' in data and data['phenotypicFeatures']:
                    for feature in data['phenotypicFeatures']:
                        feature_type = feature.get('type', {})
                        
                        # Validate dictionary type before identifier retrieval
                        if isinstance(feature_type, dict):
                            hpo_id = feature_type.get('id')
                            if hpo_id and str(hpo_id).startswith('HP:'):
                                hpo_ids.append(hpo_id)
                
                # 5. File serialization and status reporting
                if hpo_ids:
                    base_filename = os.path.splitext(filename)[0]
                    output_filename = f"{base_filename}.hpos.txt"
                    output_file_path = os.path.join(abs_output_path, output_filename)
                    
                    with open(output_file_path, 'w', encoding='utf-8') as f_out:
                        f_out.write('\n'.join(hpo_ids) + '\n')
                    
                    print(f"Success: {filename} -> {output_filename} ({len(hpo_ids)} HPO terms identified)")
                    processed_count += 1
                else:
                    print(f"Warning: {filename} skipped (No valid HPO identifiers found)")

            except Exception as e:
                print(f"Failure: An exception occurred while processing {filename}: {e}")

    print("-" * 60)
    print(f"Batch processing completed. Total files generated: {processed_count}")

if __name__ == "__main__":
    # ================= USER CONFIGURATION AREA =================
    # Define relative or absolute paths below.
    # Relative paths are calculated from the location of this script.
    
    USER_INPUT_DIR = "./raw_phenotype"
    USER_OUTPUT_DIR = "./hpos"
    # ===========================================================
    
    extract_hpo_terms(USER_INPUT_DIR, USER_OUTPUT_DIR)