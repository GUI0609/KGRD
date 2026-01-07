#!/usr/bin/env bash
set -euo pipefail

########################################
# Initialize Conda
########################################
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "ERROR: conda is not in PATH. Please ensure conda is installed and available."
    exit 1
fi

########################################
# Helper: Kill process by script path pattern
########################################
kill_if_running() {
    local pattern="$1"
    local pids
    pids=$(pgrep -f "$pattern" || true)

    if [[ -n "${pids}" ]]; then
        echo "Found running processes: $pattern (PID: $pids). Stopping..."
        kill ${pids} || true
        sleep 2

        pids=$(pgrep -f "$pattern" || true)
        if [[ -n "${pids}" ]]; then
            echo "Forcing kill -9: $pids"
            kill -9 ${pids} || true
        fi
    else
        echo "No running process found for: $pattern"
    fi
}

########################################
# Helper: Kill process using a specific port
########################################
kill_port_if_used() {
    local port="$1"
    local pids

    pids=$(lsof -ti :"$port" || true)
    if [[ -n "${pids}" ]]; then
        echo "Port ${port} is occupied by PID: ${pids}. Releasing..."
        kill ${pids} || true
        sleep 2

        pids=$(lsof -ti :"$port" || true)
        if [[ -n "${pids}" ]]; then
            echo "Port ${port} still occupied. Forcing kill -9: ${pids}"
            kill -9 ${pids} || true
        fi
    else
        echo "Port ${port} is free."
    fi
}

########################################
# Helper: Start a service
########################################
start_service() {
    local env_name="$1"
    local script_path="$2"
    local work_dir="$3"
    local log_file="$4"

    echo "-------------------------------"
    echo "Starting Service:"
    echo "  Conda Env: $env_name"
    echo "  Script:    $script_path"
    echo "  Directory: $work_dir"
    echo "  Log:       $log_file"

    conda activate "$env_name"
    cd "$work_dir"

    nohup python "$script_path" > "$log_file" 2>&1 &
    echo "Background startup complete. PID: $!"
}

########################################
# Helper: Restart a single service
########################################
restart_one() {
    local name="$1"
    local env="$2"
    local script="$3"
    local dir="$4"
    local log="$5"
    local port="$6"

    echo "==============================="
    echo "Restarting service: $name"
    echo "==============================="
    
    # 1. Clear port
    kill_port_if_used "$port"
    # 2. Kill script process
    kill_if_running "$script"
    # 3. Start service
    start_service "$env" "$script" "$dir" "$log"
    
    echo ">>> $name restart complete (Port: $port)"
}

########################################
# Path Configuration
########################################
# Base directory for utilities

BASE_DIR="/share/gguilin/rd-project/KGRD/src/KGRD_framework/utils"
# 1. Doc2Hpo
DOC2HPO_DIR="${BASE_DIR}/Doc2Hpo2.0/back-end"
DOC2HPO_SCRIPT="${DOC2HPO_DIR}/app.py"
DOC2HPO_LOG="${DOC2HPO_DIR}/app.out"

# 2. Gene-Disease Prediction 
GD_DIR="${BASE_DIR}/RDLinker-att"
GD_SCRIPT="${GD_DIR}/gene_disease_prediction_api.py"
GD_LOG="${GD_DIR}/gene_disease_prediction_api.out"

# 3. Query In KG API
QUERY_IN_KG_API_DIR="${BASE_DIR}/QUERY_IN_KG"
QUERY_IN_KG_API_SCRIPT="${QUERY_IN_KG_API_DIR}/QUERY_IN_KG_API.py"
QUERY_IN_KG_API_LOG="${QUERY_IN_KG_API_DIR}/QUERY_IN_KG_API.out"

# 4. Patient Matcher
PM_DIR="${BASE_DIR}/PATIENT_MATCHER"
PM_SCRIPT="${PM_DIR}/PATIENT_MATCHER.py"
PM_LOG="${PM_DIR}/PATIENT_MATCHER.out"

# 5. Output Directory
OUTPUT_DIR="${BASE_DIR}/output"

########################################
# Port Configuration
########################################
GD_PORT=8092
DOC2HPO_PORT=5010
QUERY_IN_KG_API_PORT=8194
PM_PORT=6006

########################################
# Kill all script processes
########################################
kill_all() {
    kill_if_running "$DOC2HPO_SCRIPT"
    kill_if_running "$QUERY_IN_KG_API_SCRIPT"
    kill_if_running "$GD_SCRIPT"
    kill_if_running "$PM_SCRIPT"
}

########################################
# Main Entry Point
########################################
target="${1:-all}"

case "$target" in
    stop)
        echo "==============================="
        echo "Stopping all services..."
        kill_all
        echo "Done."
        echo "==============================="
        ;;

    # -------------------------------------
    # Individual Service Restarts
    # -------------------------------------
    gd|GD)
        restart_one "GeneDisease" "GCN" "$GD_SCRIPT" "$GD_DIR" "$GD_LOG" "$GD_PORT"
        ;;

    doc2hpo|doc)
        restart_one "Doc2Hpo" "agent" "$DOC2HPO_SCRIPT" "$DOC2HPO_DIR" "$DOC2HPO_LOG" "$DOC2HPO_PORT"
        ;;

    kg|query_kg)
        restart_one "QueryKG" "agent" "$QUERY_IN_KG_API_SCRIPT" "$QUERY_IN_KG_API_DIR" "$QUERY_IN_KG_API_LOG" "$QUERY_IN_KG_API_PORT"
        ;;

    pm|matcher)
        restart_one "PatientMatcher" "agent" "$PM_SCRIPT" "$PM_DIR" "$PM_LOG" "$PM_PORT"
        ;;

    # -------------------------------------
    # Restart All (Default)
    # -------------------------------------
    all|restart)
        echo "==============================="
        echo "Restarting ALL services"

        echo "Cleaning output directory: ${OUTPUT_DIR}"
        mkdir -p "${OUTPUT_DIR}"
        rm -rf "${OUTPUT_DIR}"/*

        # First, release all ports
        kill_port_if_used "$GD_PORT"
        kill_port_if_used "$DOC2HPO_PORT"
        kill_port_if_used "$QUERY_IN_KG_API_PORT"
        kill_port_if_used "$PM_PORT"

        # Safe fallback kill
        kill_all

        # Start everything
        start_service "GCN"   "$GD_SCRIPT"              "$GD_DIR"              "$GD_LOG"
        start_service "agent" "$DOC2HPO_SCRIPT"         "$DOC2HPO_DIR"         "$DOC2HPO_LOG"
        start_service "agent" "$QUERY_IN_KG_API_SCRIPT" "$QUERY_IN_KG_API_DIR" "$QUERY_IN_KG_API_LOG"
        start_service "agent" "$PM_SCRIPT"              "$PM_DIR"              "$PM_LOG"

        echo "==============================="
        echo "All services have been restarted"
        echo "==============================="
        ;;

    *)
        echo "Invalid parameter."
        echo "Usage: $0 [option]"
        echo "Options:"
        echo "  all (or empty)  : Restart all services and clear the output directory"
        echo "  stop            : Stop all services"
        echo "  gd              : Restart Gene Disease API only"
        echo "  doc             : Restart Doc2Hpo API only"
        echo "  kg              : Restart Query KG API only"
        echo "  pm              : Restart Patient Matcher only"
        exit 1
        ;;
esac