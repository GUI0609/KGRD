#!/usr/bin/env bash
set -euo pipefail

START_TS=$(date +%s)

# Function to format duration in hh:mm:ss
fmt_dur() {
  local s=$1
  printf "%02d:%02d:%02d" $((s/3600)) $(((s%3600)/60)) $((s%60))
}

########################################
# Input Validation
########################################
usage() {
  echo "Usage: $0 <SAMPLE_ID>"
  exit 1
}

[[ $# -lt 1 ]] && usage
SAMPLE_ID="$1"

########################################
# User Configuration
########################################
ASSEMBLY="hg19"
EXOMISER_VERSION="14.0.0"

# Paths
BASE="YOUR_BASE_DIR"
PUBLIC_TESTDATA="${BASE}/public_run/testdata_dir"

# LIRICAL Configuration
LIRICAL_JAR="${BASE}/lirical_run/input_dir/lirical-cli-2.2.0/lirical-cli-2.2.0.jar"
LIRICAL_EXOMISER_DIR="${BASE}/lirical_run/input_dir/2406_hg19"
LIRICAL_OUTPUT_DIR="${BASE}/public_run/output_dir/lirical_output"
PHENOPACKET="${PUBLIC_TESTDATA}/phenopackets/${SAMPLE_ID}.json"
VCF="${PUBLIC_TESTDATA}/vcf/${SAMPLE_ID}.vcf"

# AI-MARRVEL Configuration
AIM_REF_DIR="${BASE}/temp_download_ai_marrvel"
AIM_MAIN_NF="${AIM_REF_DIR}/AI_MARRVEL/main.nf"
AIM_HPO="${PUBLIC_TESTDATA}/hpos/${SAMPLE_ID}.hpos.txt"
AIM_OUTDIR="${BASE}/public_run/output_dir/ai_marrvel_output"
AIM_STORE="${BASE}/ai_marrvel_run/store_dir"
AIM_REF_VER="${ASSEMBLY}"

# Exomiser Configuration (via Pheval runner)
EXO_WORKDIR="${BASE}/exomiser_run"
EXO_INPUT_DIR="${EXO_WORKDIR}/input_dir"
EXO_TESTDATA_DIR="${PUBLIC_TESTDATA}"
EXO_OUTPUT_DIR="${BASE}/public_run/output_dir/exomiser_output"

# Logging
LOG_DIR="${BASE}/public_run/output_dir/logs/${SAMPLE_ID}"
mkdir -p "${LOG_DIR}"

########################################
# Helper Functions
########################################
# Activate Conda environment in non-interactive mode
conda_activate() {
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$1"
  else
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate "$1"
  fi
}

pids=()
names=()

run_lirical() {
  (
    conda_activate "pheval2"
    java -jar "${LIRICAL_JAR}" phenopacket \
      --assembly "${ASSEMBLY}" \
      --exomiser-hg19-dir "${LIRICAL_EXOMISER_DIR}" \
      --phenopacket "${PHENOPACKET}" \
      --vcf "${VCF}" \
      --prefix "${SAMPLE_ID}" \
      --output-format tsv \
      --output-directory "${LIRICAL_OUTPUT_DIR}"
  ) >"${LOG_DIR}/lirical_${SAMPLE_ID}.log" 2>&1
}

run_ai_marrvel() {
  (
    conda_activate "pheval-ai-marrvel"
    nextflow run "${AIM_MAIN_NF}" \
      -profile debug \
      --ref_dir "${AIM_REF_DIR}" \
      --input_vcf "${VCF}" \
      --input_hpo "${AIM_HPO}" \
      --outdir "${AIM_OUTDIR}" \
      --storedir "${AIM_STORE}" \
      --run_id "${SAMPLE_ID}" \
      --ref_ver "${AIM_REF_VER}"
  ) >"${LOG_DIR}/ai_marrvel_${SAMPLE_ID}.log" 2>&1
}

run_exomiser() {
  (
    conda_activate "autogen"
    cd "${EXO_WORKDIR}"
    pheval run \
      --input-dir   "${EXO_INPUT_DIR}" \
      --testdata-dir "${EXO_TESTDATA_DIR}" \
      --output-dir  "${EXO_OUTPUT_DIR}" \
      --runner exomiserphevalrunner \
      --version "${EXOMISER_VERSION}"
  ) >"${LOG_DIR}/exomiser_${SAMPLE_ID}.log" 2>&1
}

########################################
# Pipeline Execution
########################################
echo "[$(date '+%F %T')] Starting pipelines for sample ${SAMPLE_ID}"

echo "[$(date '+%F %T')] >> Running step: generate_hpos.py"
python /share/gguilin/rd-project/pheval/public_run/resourse/generate_hpos.py

echo "[$(date '+%F %T')] >> Running step: generate_json.py"
python /share/gguilin/rd-project/pheval/public_run/resourse/generate_json.py "${SAMPLE_ID}"

echo "[$(date '+%F %T')] >> Launching parallel task: LIRICAL"
(run_lirical)    & pids+=($!); names+=("LIRICAL")

echo "[$(date '+%F %T')] >> Launching parallel task: AI-MARRVEL"
(run_ai_marrvel) & pids+=($!); names+=("AI-MARRVEL")

echo "[$(date '+%F %T')] >> Launching parallel task: Exomiser"
(run_exomiser)   & pids+=($!); names+=("Exomiser")

# Wait for completion and aggregate exit statuses
exit_code=0
for i in "${!pids[@]}"; do
  pid=${pids[$i]}
  name=${names[$i]}
  if wait "$pid"; then
    echo "[$(date '+%F %T')] ${name} finished successfully."
  else
    echo "[$(date '+%F %T')] ${name} failed. Check log: ${LOG_DIR}/$(echo "${name}" | tr '[:upper:]' '[:lower:]')_${SAMPLE_ID}.log"
    exit_code=1
  fi
done

########################################
# Post-processing: Gene Extraction and Timing
########################################
echo "[$(date '+%F %T')] Starting *_extract_gene.py scripts ..."

conda_activate "pheval2"

SCRIPTS=(
  "${AIM_OUTDIR}/ai_marrvel_extract_gene.py"
  "${EXO_OUTPUT_DIR}/exomiser_extract_gene.py"
  "${LIRICAL_OUTPUT_DIR}/lirical_extract_gene.py"
)

for py in "${SCRIPTS[@]}"; do
  start_ts=$(date +%s)
  echo "[$(date '+%F %T')] Running ${py} ..."
  python "${py}" >"${LOG_DIR}/$(basename "${py}" .py)_${SAMPLE_ID}.log" 2>&1
  rc=$?
  end_ts=$(date +%s)
  dur=$((end_ts - start_ts))
  if [[ $rc -eq 0 ]]; then
    echo "[$(date '+%F %T')] ${py} finished successfully. Elapsed: $(fmt_dur ${dur})"
  else
    echo "[$(date '+%F %T')] ${py} failed. Elapsed: $(fmt_dur ${dur}). Check log: ${LOG_DIR}/$(basename "${py}" .py)_${SAMPLE_ID}.log"
    exit_code=1
  fi
done

echo "[$(date '+%F %T')] All tasks completed with status ${exit_code}."

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo "[$(date '+%F %T')] Total elapsed (hh:mm:ss): $(fmt_dur ${ELAPSED})"

########################################
# Workspace Cleanup
########################################
echo "[$(date '+%F %T')] Cleaning up workspace..."
rm -f "${PUBLIC_TESTDATA}/hpos/"*
rm -f "${PUBLIC_TESTDATA}/phenopackets/"*
rm -f "${PUBLIC_TESTDATA}/raw_phenotype/"*
rm -f "${PUBLIC_TESTDATA}/vcf/"*
echo "[$(date '+%F %T')] Workspace cleaned."

exit "${exit_code}"