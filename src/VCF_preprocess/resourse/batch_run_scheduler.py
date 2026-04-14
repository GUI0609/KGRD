import csv
import os
import shutil
import subprocess
import time
import sys

def run_batch_pipeline():
    # ================= USER CONFIGURATION AREA =================
    # Paths can be absolute or relative to the script's directory.
    
    # 1. Path to the CSV file containing the list of samples
    CSV_FILE_PATH = "xmu_sample_test/xmu_rare.csv"
    
    # 2. Source directories for raw genomic and phenotypic data
    SOURCE_VCF_DIR = "../xmu_vcf"
    SOURCE_YAML_DIR = "../phenotype_yamls"
    
    # 3. Target working directories for analysis
    WORK_VCF_DIR = "./testdata_dir/vcf"
    WORK_YAML_DIR = "./testdata_dir/raw_phenotype"
    
    # 4. Path to the shell script executing the analysis pipeline
    PIPELINE_SCRIPT = "./run_rd_pipelines.sh"
    # ===========================================================

    # Resolve absolute paths to ensure reliability
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    def get_abs_path(path):
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(base_dir, path))

    csv_path = get_abs_path(CSV_FILE_PATH)
    src_vcf_dir = get_abs_path(SOURCE_VCF_DIR)
    src_yaml_dir = get_abs_path(SOURCE_YAML_DIR)
    target_vcf_dir = get_abs_path(WORK_VCF_DIR)
    target_yaml_dir = get_abs_path(WORK_YAML_DIR)
    script_path = get_abs_path(PIPELINE_SCRIPT)

    if not os.path.exists(csv_path):
        print(f"Critical Error: CSV file not found at {csv_path}")
        return

    # Ensure output directories exist
    os.makedirs(target_vcf_dir, exist_ok=True)
    os.makedirs(target_yaml_dir, exist_ok=True)

    print("Initiating batch processing pipeline...")
    print(f"Sample manifest: {csv_path}")

    success_count = 0
    fail_count = 0
    
    # Parse sample list
    valid_rows = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row
            for row in reader:
                if row and row[0].strip():
                    valid_rows.append(row)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    total_samples = len(valid_rows)
    print(f"Preprocessing complete: {total_samples} valid samples identified for processing.")

    # Iterate through each sample
    for index, row in enumerate(valid_rows, start=1):
        sample_id = row[0].strip()
        
        print(f"\n" + "="*70)
        print(f"Processing Sample: {sample_id} [{index}/{total_samples}]")
        print(f"="*70)

        src_vcf = os.path.join(src_vcf_dir, f"{sample_id}.vcf")
        src_yml = os.path.join(src_yaml_dir, f"{sample_id}.yml")
        dst_vcf = os.path.join(target_vcf_dir, f"{sample_id}.vcf")
        dst_yml = os.path.join(target_yaml_dir, f"{sample_id}.yml")

        # Validate existence of source files
        if not os.path.exists(src_vcf):
            print(f"Warning: Source VCF missing -> {src_vcf}")
            fail_count += 1
            continue
        if not os.path.exists(src_yml):
            print(f"Warning: Source YAML missing -> {src_yml}")
            fail_count += 1
            continue

        try:
            print(f"Step [1/3]: Synchronizing source files...")
            shutil.copy2(src_vcf, dst_vcf)
            shutil.copy2(src_yml, dst_yml)
            print("Files successfully staged.")

            print(f"Step [2/3]: Executing Bash pipeline (Streaming Log Output):")
            print(f"----------------------------------------------------------------------")
            
            start_time = time.time()

            # Execute pipeline using login shell to ensure environment consistency (e.g., conda)
            process = subprocess.Popen(
                ["bash", "-l", script_path, sample_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, 
                text=True,
                bufsize=1 
            )

            # Stream output in real-time
            for line in process.stdout:
                print(f"   [Pipeline]: {line}", end='')
                sys.stdout.flush() 

            process.wait()
            
            if process.returncode == 0:
                execution_time = time.time() - start_time
                print(f"----------------------------------------------------------------------")
                print(f"Step [3/3]: Sample {sample_id} processed successfully. Runtime: {execution_time:.2f}s")
                success_count += 1
            else:
                print(f"----------------------------------------------------------------------")
                print(f"Failure: Pipeline terminated with exit code {process.returncode} for sample {sample_id}")
                fail_count += 1

        except Exception as e:
            print(f"Exception: An error occurred during Python execution: {e}")
            fail_count += 1

    # Final summary report
    print(f"\n" + "="*70)
    print(f"Batch Processing Final Report")
    print(f"Successful Operations: {success_count}")
    print(f"Failed Operations:     {fail_count}")
    print(f"Completion Status:     {'Success' if fail_count == 0 else 'Completed with errors'}")
    print(f"="*70)

if __name__ == "__main__":
    run_batch_pipeline()