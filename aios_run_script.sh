#!/bin/bash

set -euo pipefail

LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"

export AIOS_MODEL="${AIOS_MODEL:-models/tiiuae-falcon-7b-Q4_K_M.gguf}"
export AIOS_PYVER="${AIOS_PYVER:-3.14}"
export AIOS_OUTPUT_FOLDER="./results/"

usage() {
        cat <<EOF
Usage: $0 [options] [script]

Options:
  --script PATH              Python script to run
  --model PATH               Model path
  --runs COUNT               Number of runs
  --tokens COUNT             Number of tokens
  --output-folder PATH       Output folder
  --output-filename NAME     Output filename without .json
  --pyver VERSION            Python version metadata
  -h, --help                 Show this help

The positional [script] argument is still supported.
EOF
}

AIOS_SCRIPT=""
AIOS_RUNS="${AIOS_RUNS:-5}"
AIOS_TOKENS="${AIOS_TOKENS:-200}"

AIOS_OUTPUT_FILENAME="${AIOS_OUTPUT_FILENAME:-baseline}"

while [[ $# -gt 0 ]]; do
        case "$1" in
                --script)
                        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
                        AIOS_SCRIPT="$2"
                        shift 2
                        ;;
                --model)
                        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
                        AIOS_MODEL="$2"
                        shift 2
                        ;;
                --runs)
                        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
                        AIOS_RUNS="$2"
                        shift 2
                        ;;
                --tokens)
                        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
                        AIOS_TOKENS="$2"
                        shift 2
                        ;;
                --output-folder)
                        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
                        AIOS_OUTPUT_FOLDER="$2"
                        shift 2
                        ;;
                --output-filename)
                        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
                        AIOS_OUTPUT_FILENAME="$2"
                        shift 2
                        ;;
                --pyver)
                        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 1; }
                        AIOS_PYVER="$2"
                        shift 2
                        ;;
                -h|--help)
                        usage
                        exit 0
                        ;;
                --)
                        shift
                        break
                        ;;
                -*)
                        echo "Unknown option: $1" >&2
                        usage >&2
                        exit 1
                        ;;
                *)
                        if [[ -z "${AIOS_SCRIPT}" ]]; then
                                AIOS_SCRIPT="$1"
                                shift
                        else
                                echo "Unexpected positional argument: $1" >&2
                                usage >&2
                                exit 1
                        fi
                        ;;
        esac
done

if [[ $# -gt 0 ]]; then
        if [[ -z "${AIOS_SCRIPT}" ]]; then
                AIOS_SCRIPT="$1"
                shift
        fi

        if [[ $# -gt 0 ]]; then
                echo "Unexpected positional argument: $1" >&2
                usage >&2
                exit 1
        fi
fi

AIOS_SCRIPT="${AIOS_SCRIPT:-validation/baseline.py}"

if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root" 1>&2
   exit 1
fi

echo "Running AIOS $0"

OLD_NMI_WATCHDOG=$(cat /proc/sys/kernel/nmi_watchdog)
OLD_PERF_EVENT_PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid)

echo "0" | tee /proc/sys/kernel/nmi_watchdog 
echo "0" | tee /proc/sys/kernel/perf_event_paranoid 

restore() {
        echo "${OLD_NMI_WATCHDOG}"        | tee /proc/sys/kernel/nmi_watchdog 
        echo "${OLD_PERF_EVENT_PARANOID}" | tee /proc/sys/kernel/perf_event_paranoid 
}

trap restore EXIT SIGINT SIGTERM

python "${AIOS_SCRIPT}" \
        --model  "${AIOS_MODEL}" \
        --runs   "${AIOS_RUNS}" \
        --tokens "${AIOS_TOKENS}" \
        --output "${AIOS_OUTPUT_FOLDER}/${AIOS_OUTPUT_FILENAME}.json" \
        --verbose

