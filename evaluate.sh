#!/bin/bash

# Default values for parameters
ALGORITHM=""
MODEL_NAME="blip2"
VQAV2_EVAL_ANNOTATIONS_JSON_PATH="data/vqav2_eval_annotations.json"
DEVICE="0"
MAX_GENERATION_LENGTH="5"
NUM_BEAMS="3"
LENGTH_PENALTY="-2.0"
NUM_SHOTS="2"
ALPHA1="0.00392156862" # 1/255
EPSILON="0.062745098" # 16/255
ITERS="200"
ALPHA2="0.01"
FRACTION="0.01"
BASE_DIR="./"
PROMPT_NUM="1"
TARGET_TEXT_FROM_ARG="" # To store target if passed directly
PROMPT_FILE="" # To store path to prompt.txt

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --algorithm=*) ALGORITHM="${1#*=}" ;;
        --model_name=*) MODEL_NAME="${1#*=}" ;;
        --vqav2_eval_annotations_json_path=*) VQAV2_EVAL_ANNOTATIONS_JSON_PATH="${1#*=}" ;;
        --device=*) DEVICE="${1#*=}" ;;
        --max_generation_length=*) MAX_GENERATION_LENGTH="${1#*=}" ;;
        --num_beams=*) NUM_BEAMS="${1#*=}" ;;
        --length_penalty=*) LENGTH_PENALTY="${1#*=}" ;;
        --num_shots=*) NUM_SHOTS="${1#*=}" ;;
        --alpha1=*) ALPHA1="${1#*=}" ;;
        --epsilon=*) EPSILON="${1#*=}" ;;
        --iters=*) ITERS="${1#*=}" ;;
        --alpha2=*) ALPHA2="${1#*=}" ;;
        --fraction=*) FRACTION="${1#*=}" ;;
        --base_dir=*) BASE_DIR="${1#*=}" ;;
        --prompt_num=*) PROMPT_NUM="${1#*=}" ;;
        --target=*) TARGET_TEXT_FROM_ARG="${1#*=}" ;;
        --prompt_file=*) PROMPT_FILE="${1#*=}" ;;
        # For space-separated arguments (e.g., --param value)
        --algorithm) ALGORITHM="$2"; shift ;;
        --model_name) MODEL_NAME="$2"; shift ;;
        --vqav2_eval_annotations_json_path) VQAV2_EVAL_ANNOTATIONS_JSON_PATH="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --max_generation_length) MAX_GENERATION_LENGTH="$2"; shift ;;
        --num_beams) NUM_BEAMS="$2"; shift ;;
        --length_penalty) LENGTH_PENALTY="$2"; shift ;;
        --num_shots) NUM_SHOTS="$2"; shift ;;
        --alpha1) ALPHA1="$2"; shift ;;
        --epsilon) EPSILON="$2"; shift ;;
        --iters) ITERS="$2"; shift ;;
        --alpha2) ALPHA2="$2"; shift ;;
        --fraction) FRACTION="$2"; shift ;;
        --base_dir) BASE_DIR="$2"; shift ;;
        --prompt_num) PROMPT_NUM="$2"; shift ;;
        --target) TARGET_TEXT_FROM_ARG="$2"; shift ;;
        --prompt_file) PROMPT_FILE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate algorithm choice
if [[ -z "$ALGORITHM" ]]; then
    echo "Error: --algorithm parameter is required."
    echo "Usage: ./run_algorithm.sh --algorithm=<cropa|duap|init> [--prompt_file=/path/to/prompt.txt | --target=\"your_target_text\"] [other_parameters...]"
    exit 1
fi

PYTHON_SCRIPT=""
METHOD_PARAM=""
DEFAULT_TARGET_TEXT=""

case "$ALGORITHM" in
    "cropa")
        PYTHON_SCRIPT="main.py"
        METHOD_PARAM="cropa"
        DEFAULT_TARGET_TEXT="bomb"
        ;;
    "duap")
        PYTHON_SCRIPT="main_with_duap.py"
        METHOD_PARAM="cropa"
        DEFAULT_TARGET_TEXT="bomb"
        ;;
    "init")
        PYTHON_SCRIPT="noise_init/main_with_init.py"
        METHOD_PARAM="cropa"
        DEFAULT_TARGET_TEXT="bomb"
        ;;
    *)
        echo "Invalid algorithm selected: $ALGORITHM"
        echo "Supported algorithms are: cropa, duap, init"
        exit 1
        ;;
esac

# Determine target text source
TARGET_TEXTS=()
if [[ -n "$PROMPT_FILE" ]]; then
    if [[ -f "$PROMPT_FILE" ]]; then
        echo "Reading target texts from $PROMPT_FILE..."
        while IFS= read -r line || [[ -n "$line" ]]; do
            # Skip empty lines and lines starting with #
            if [[ -n "$line" && ! "$line" =~ ^\# ]]; then
                TARGET_TEXTS+=("$line")
            fi
        done < "$PROMPT_FILE"
        if [[ ${#TARGET_TEXTS[@]} -eq 0 ]]; then
            echo "Warning: No valid target texts found in $PROMPT_FILE. Using default target."
            TARGET_TEXTS+=("$DEFAULT_TARGET_TEXT")
        fi
    else
        echo "Error: Prompt file '$PROMPT_FILE' not found. Exiting."
        exit 1
    fi
elif [[ -n "$TARGET_TEXT_FROM_ARG" ]]; then
    TARGET_TEXTS+=("$TARGET_TEXT_FROM_ARG")
else
    # Fallback to default if neither file nor direct target is provided
    TARGET_TEXTS+=("$DEFAULT_TARGET_TEXT")
fi

echo "Running $PYTHON_SCRIPT with the following parameters:"
echo "  Algorithm: $ALGORITHM"
echo "  Model Name: $MODEL_NAME"
echo "  VQAV2 Annotations Path: $VQAV2_EVAL_ANNOTATIONS_JSON_PATH"
echo "  Device: $DEVICE"
echo "  Max Generation Length: $MAX_GENERATION_LENGTH"
echo "  Num Beams: $NUM_BEAMS"
echo "  Length Penalty: $LENGTH_PENALTY"
echo "  Num Shots: $NUM_SHOTS"
echo "  Alpha1: $ALPHA1"
echo "  Epsilon: $EPSILON"
echo "  Iters: $ITERS"
echo "  Alpha2: $ALPHA2"
echo "  Fraction: $FRACTION"
echo "  Base Directory: $BASE_DIR"
echo "  Prompt Num: $PROMPT_NUM"
echo "  Method Parameter: $METHOD_PARAM"


# Loop through each target text and execute the Python script
for CURRENT_TARGET_TEXT in "${TARGET_TEXTS[@]}"; do
    echo "--------------------------------------------------------"
    echo "Executing with Target Text: $CURRENT_TARGET_TEXT"
    echo "--------------------------------------------------------"

    python "$PYTHON_SCRIPT" \
        --model_name "$MODEL_NAME" \
        --vqav2_eval_annotations_json_path "$VQAV2_EVAL_ANNOTATIONS_JSON_PATH" \
        --device "$DEVICE" \
        --max_generation_length "$MAX_GENERATION_LENGTH" \
        --num_beams "$NUM_BEAMS" \
        --length_penalty "$LENGTH_PENALTY" \
        --num_shots "$NUM_SHOTS" \
        --alpha1 "$ALPHA1" \
        --epsilon "$EPSILON" \
        --iters "$ITERS" \
        --alpha2 "$ALPHA2" \
        --fraction "$FRACTION" \
        --base_dir "$BASE_DIR" \
        --prompt_num "$PROMPT_NUM" \
        --target "$CURRENT_TARGET_TEXT" \
        --method "$METHOD_PARAM"
done

echo "Script execution complete for all specified target texts."