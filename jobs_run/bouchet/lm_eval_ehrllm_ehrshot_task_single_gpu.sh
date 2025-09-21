#!/bin/bash

# Default values
MODEL_NAME=""
MAX_MODEL_LEN=""
LIMIT=""
BATCH_SIZE="auto"
GPU_MEMORY_UTILIZATION="0.9"
LOG_SAMPLES=false
THINK_END_TOKEN="</think>"
DEBUG=false
BACKEND="vllm"

# Global variables for inference configuration
TENSOR_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=1
DTYPE="bfloat16"

# GPU_MEMORY_UTILIZATION and BATCH_SIZE are now set from command line arguments

# Output directory configuration
RESULTS_ROOT_DIR="/home/yl2342/project_pi_hx235/yl2342/bids-lm-evaluation/results"

# Task include path configuration
INCLUDE_PATH="/home/yl2342/project_pi_hx235/yl2342/bids-lm-evaluation/bids_tasks/ehr_llm"


# Function to display usage
usage() {
    echo "Usage: $0 --model_name <model> --max_model_len <length> [OPTIONS]"
    echo ""
    echo "Required arguments:"
    echo "  --model_name <model>        Model name/path"
    echo "                              meta-llama: meta-llama/Llama-3.2-3B-Instruct, meta-llama/Llama-3.1-8B-Instruct"
    echo "                              Qwen: Qwen/Qwen3-1.7B, Qwen/Qwen3-4B, Qwen/Qwen3-8B, Qwen/Qwen3-14B"
    echo "  --max_model_len <length>    Maximum model length (e.g., 8192, 32768)"
    echo ""
    echo "Optional arguments:"
    echo "  --backend <backend>         Backend to use: 'vllm' or 'hf' (default: vllm)"
    echo "  --limit <number>            Number of samples to evaluate per task"
    echo "                              (default: all samples)"
    echo "  --batch_size <size>         Batch size for evaluation: positive integer or 'auto' (default: auto)"
    echo "  --gpu_memory_util <ratio>   GPU memory utilization ratio 0.0-1.0 (default: 0.8)"
    echo "  --think_end_token <token>   End token for thinking models (default: </think>)"
    echo "  --log_samples               Log individual sample outputs for debugging"
    echo "  --debug                     Save results to debug directory structure"
    echo "  --help                      Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Standard models with vLLM backend"
    echo "  $0 --model_name meta-llama/Llama-3.2-3B-Instruct --max_model_len 8192"
    echo "  $0 --model_name meta-llama/Llama-3.1-8B-Instruct --max_model_len 8192 --limit 1000"
    echo ""
    echo "  # Standard models with HuggingFace backend"
    echo "  $0 --model_name meta-llama/Llama-3.2-3B-Instruct --max_model_len 8192 --backend hf"
    echo ""
    echo "  # Thinking models (automatically includes thinking parameters)"
    echo "  $0 --model_name Qwen/Qwen3-1.7B --max_model_len 8192"
    echo "  $0 --model_name Qwen/Qwen3-4B --max_model_len 32768 --limit 100 --backend hf"
    echo ""
    echo "  # Custom think end token"
    echo "  $0 --model_name Qwen/Qwen3-1.7B --max_model_len 8192 --think_end_token '</reasoning>'"
    echo ""
    echo "  # Debugging"
    echo "  $0 --model_name Qwen/Qwen3-1.7B --max_model_len 8192 --limit 10 --log_samples"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --max_model_len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gpu_memory_util)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --think_end_token)
            THINK_END_TOKEN="$2"
            shift 2
            ;;
        --log_samples)
            LOG_SAMPLES=true
            shift 1
            ;;
        --debug)
            DEBUG=true
            shift 1
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            echo ""
            usage
            ;;
    esac
done

# Debug: Print received arguments
echo "$(date): DEBUG - Total arguments received: $#"
echo "$(date): DEBUG - All arguments: $@"
echo "$(date): DEBUG - MODEL_NAME='$MODEL_NAME'"
echo "$(date): DEBUG - MAX_MODEL_LEN='$MAX_MODEL_LEN'"
echo "$(date): DEBUG - BACKEND='$BACKEND'"

# Validate required arguments
if [ -z "$MODEL_NAME" ]; then
    echo "Error: --model_name is required"
    echo ""
    usage
fi

if [ -z "$MAX_MODEL_LEN" ]; then
    echo "Error: --max_model_len is required"
    echo ""
    usage
fi

# Validate backend
if [ "$BACKEND" != "vllm" ] && [ "$BACKEND" != "hf" ]; then
    echo "Error: --backend must be either 'vllm' or 'hf'"
    exit 1
fi

# Validate max_model_len is a number
if ! [[ "$MAX_MODEL_LEN" =~ ^[0-9]+$ ]]; then
    echo "Error: --max_model_len must be a positive integer"
    exit 1
fi

# Validate limit is a number if provided
if [ -n "$LIMIT" ] && ! [[ "$LIMIT" =~ ^[0-9]+$ ]]; then
    echo "Error: --limit must be a positive integer"
    exit 1
fi

# Validate batch_size is a number or "auto"
if [ "$BATCH_SIZE" != "auto" ] && ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]]; then
    echo "Error: --batch_size must be a positive integer or 'auto'"
    exit 1
fi

# Validate gpu_memory_utilization is a valid float between 0 and 1
if ! [[ "$GPU_MEMORY_UTILIZATION" =~ ^0*\.?[0-9]+$ ]] || (( $(echo "$GPU_MEMORY_UTILIZATION > 1" | bc -l) )) || (( $(echo "$GPU_MEMORY_UTILIZATION <= 0" | bc -l) )); then
    echo "Error: --gpu_memory_util must be a number between 0 and 1"
    exit 1
fi

# Construct model arguments string based on backend (AFTER argument parsing)
if [ "$BACKEND" = "vllm" ]; then
    # vLLM backend configuration
    # Note: enable_thinking=True and think_end_token are automatically included for all models
    # This enables thinking/reasoning capabilities for models that support it (e.g., Qwen models)
    MODEL_ARGS="pretrained=${MODEL_NAME},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},data_parallel_size=${DATA_PARALLEL_SIZE},dtype=${DTYPE},max_model_len=${MAX_MODEL_LEN},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},enable_thinking=True,think_end_token=${THINK_END_TOKEN}"
elif [ "$BACKEND" = "hf" ]; then
    # HuggingFace backend configuration
    MODEL_ARGS="pretrained=${MODEL_NAME},dtype=${DTYPE},max_length=${MAX_MODEL_LEN},trust_remote_code=True,device_map=auto,enable_thinking=True,think_end_token=${THINK_END_TOKEN}"
fi


# Set limit argument if provided
LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

# Set log_samples argument if provided
LOG_SAMPLES_ARG=""
if [ "$LOG_SAMPLES" = true ]; then
    LOG_SAMPLES_ARG="--log_samples"
fi

# Set output directory based on debug flag
if [ "$DEBUG" = true ]; then
    OUTPUT_BASE_DIR="${RESULTS_ROOT_DIR}/debug"
    echo "$(date): DEBUG MODE: Results will be saved to debug directory: $OUTPUT_BASE_DIR"
else
    OUTPUT_BASE_DIR="${RESULTS_ROOT_DIR}/ehr_llm/ehrshot"
    echo "$(date): Results will be saved to: $OUTPUT_BASE_DIR"
fi

# environment setup
# HuggingFace cache is configured in ~/.bashrc to use scratch space

# Set Hugging Face token from the saved token file
if [ -f "$HF_HOME/token" ]; then
    export HF_TOKEN=$(cat "$HF_HOME/token")
    echo "$(date): Hugging Face token loaded from $HF_HOME/token"
else
    echo "$(date): Warning: No Hugging Face token found at $HF_HOME/token. You may need to run 'huggingface-cli login'"
fi

# PyTorch and vLLM optimization settings
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_USAGE_STATS_DISABLED=1

# Ensure torch compile cache directory exists (configured in ~/.bashrc)
mkdir -p "$TORCH_COMPILE_CACHE_DIR"

echo "$(date): Using PyTorch compile cache directory: $TORCH_COMPILE_CACHE_DIR"
echo "$(date): vLLM usage tracking disabled to avoid disk quota issues"

echo "$(date): Using Hugging Face cache directory: $HF_HOME (from ~/.bashrc)"

# Ensure directories exist
mkdir -p "$OUTPUT_BASE_DIR"
echo "$(date): Results will be saved to: $OUTPUT_BASE_DIR"

# conda environment setup
module load miniconda
conda activate bids_lm_eval 
# EHRShot benchmark evaluation
# Includes thinking model support with enable_thinking=True and think_end_token='</think>'

echo "$(date): Starting evaluation for ${MODEL_NAME} with max_model_len ${MAX_MODEL_LEN}"
echo "$(date): Model arguments: ${MODEL_ARGS}"
if [ -n "$LIMIT" ]; then
    echo "$(date): Using limit of ${LIMIT} samples per task"
else
    echo "$(date): Using all available samples"
fi

## Task 1: Inpatient tasks
echo "$(date): Starting inpatient tasks..."
lm_eval \
   --model ${BACKEND} \
   --model_args ${MODEL_ARGS} \
   --apply_chat_template \
   --include_path ${INCLUDE_PATH} \
   --tasks group_ehrshot_inpatient_tasks_gu \
   --batch_size ${BATCH_SIZE} \
   --output_path ${OUTPUT_BASE_DIR}/task_inpatient/max_len_${MAX_MODEL_LEN} \
   --metadata '{"model_name": "'${MODEL_NAME}'", "max_model_len": "'${MAX_MODEL_LEN}'", "task_name": "inpatient"}' \
   ${LIMIT_ARG} ${LOG_SAMPLES_ARG}

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "$(date): ✓ Inpatient tasks completed successfully"
else
    echo "$(date): ✗ Inpatient tasks failed with exit code $exit_code"
    exit $exit_code
fi



## Task 2: Measurement tasks (labs and vitals)
echo "$(date): Starting measurement lab tasks..."
lm_eval \
   --model ${BACKEND} \
   --model_args ${MODEL_ARGS} \
   --apply_chat_template \
   --include_path ${INCLUDE_PATH} \
   --tasks group_ehrshot_measurement_lab_tasks_gu \
   --batch_size ${BATCH_SIZE} \
   --output_path ${OUTPUT_BASE_DIR}/task_measurement/lab/max_len_${MAX_MODEL_LEN} \
   --metadata '{"model_name": "'${MODEL_NAME}'", "max_model_len": "'${MAX_MODEL_LEN}'", "task_name": "measurement_lab"}' \
   ${LIMIT_ARG} ${LOG_SAMPLES_ARG}

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "$(date): ✓ Measurement lab tasks completed successfully"
else
    echo "$(date): ✗ Measurement lab tasks failed with exit code $exit_code"
    exit $exit_code
fi

echo "$(date): Starting measurement vital tasks..."
lm_eval \
   --model ${BACKEND} \
   --model_args ${MODEL_ARGS} \
   --apply_chat_template \
   --include_path ${INCLUDE_PATH} \
   --tasks group_ehrshot_measurement_vital_tasks_gu \
   --batch_size ${BATCH_SIZE} \
   --output_path ${OUTPUT_BASE_DIR}/task_measurement/vital/max_len_${MAX_MODEL_LEN} \
   --metadata '{"model_name": "'${MODEL_NAME}'", "max_model_len": "'${MAX_MODEL_LEN}'", "task_name": "measurement_vital"}' \
   ${LIMIT_ARG} ${LOG_SAMPLES_ARG}

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "$(date): ✓ Measurement vital tasks completed successfully"
else
    echo "$(date): ✗ Measurement vital tasks failed with exit code $exit_code"
    exit $exit_code
fi

## Task 3: Diagnosis tasks (new and recurrent)
echo "$(date): Starting new diagnosis tasks..."
lm_eval \
   --model ${BACKEND} \
   --model_args ${MODEL_ARGS} \
   --apply_chat_template \
   --include_path ${INCLUDE_PATH} \
   --tasks group_ehrshot_new_diagnosis_tasks_gu \
   --batch_size ${BATCH_SIZE} \
   --output_path ${OUTPUT_BASE_DIR}/task_diagnosis/new/max_len_${MAX_MODEL_LEN} \
   --metadata '{"model_name": "'${MODEL_NAME}'", "max_model_len": "'${MAX_MODEL_LEN}'", "task_name": "diagnosis_new"}' \
   ${LIMIT_ARG} ${LOG_SAMPLES_ARG}

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "$(date): ✓ New diagnosis tasks completed successfully"
else
    echo "$(date): ✗ New diagnosis tasks failed with exit code $exit_code"
    exit $exit_code
fi

echo "$(date): Starting recurrent diagnosis tasks..."
lm_eval \
   --model ${BACKEND} \
   --model_args ${MODEL_ARGS} \
   --apply_chat_template \
   --include_path ${INCLUDE_PATH} \
   --tasks group_ehrshot_recurrent_diagnosis_tasks_gu \
   --batch_size ${BATCH_SIZE} \
   --output_path ${OUTPUT_BASE_DIR}/task_diagnosis/recurrent/max_len_${MAX_MODEL_LEN} \
   --metadata '{"model_name": "'${MODEL_NAME}'", "max_model_len": "'${MAX_MODEL_LEN}'", "task_name": "diagnosis_recurrent"}' \
   ${LIMIT_ARG} ${LOG_SAMPLES_ARG}

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "$(date): ✓ Recurrent diagnosis tasks completed successfully"
else
    echo "$(date): ✗ Recurrent diagnosis tasks failed with exit code $exit_code"
    exit $exit_code
fi

echo "$(date): All evaluation tasks completed!"

# Cleanup: deactivate conda environment and unload miniconda module
echo "$(date): Cleaning up environment..."
conda deactivate
module unload miniconda
echo "$(date): Environment cleanup completed"