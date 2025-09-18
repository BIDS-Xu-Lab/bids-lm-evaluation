# EHRShot Evaluation Scripts

> **YCRC Misha Cluster Only**  
> These scripts are designed to run on the **YCRC Misha cluster** specifically.  
> EHRShot data is located at: `/gpfs/radev/pi/xu_hua/shared/ehr_llm/ehrshot/visit_oriented_ehr/`

This directory contains scripts for evaluating language models on the EHRShot benchmark using the lm-evaluation-harness framework.

## Scripts

- **`lm_eval_ehrllm_ehrshot_task_slurm.sh`**: SLURM-based evaluation script for cluster submission
- **`lm_eval_ehrllm_ehrshot_task.sh`**: Interactive evaluation script for direct execution on compute nodes

Both scripts are fully aligned and use the same task configurations, conda environment (`bids_lm_eval`), and support thinking models.

## Quick Start

### SLURM Script (Recommended)

```bash
# Basic usage
sbatch lm_eval_ehrllm_ehrshot_task_vllm_slurm.sh --model_name meta-llama/Llama-3.2-3B-Instruct --max_model_len 8192

# With thinking models
sbatch lm_eval_ehrllm_ehrshot_task_vllm_slurm.sh --model_name Qwen/Qwen3-4B --max_model_len 40

# Testing with limited samples
sbatch lm_eval_ehrllm_ehrshot_task_slurm.sh --model_name meta-llama/Llama-3.1-8B-Instruct --max_model_len 8192 --limit 100
```

### Interactive Script

```bash
# Basic usage (run on compute node)
./lm_eval_ehrllm_ehrshot_task.sh --model_name meta-llama/Llama-3.2-3B-Instruct --max_model_len 8192

# With debugging
./lm_eval_ehrllm_ehrshot_task.sh --model_name Qwen/Qwen3-1.7B --max_model_len 8192 --limit 10 --log_samples
sbatch lm_eval_ehrllm_ehrshot_task_vllm_slurm.sh --model_name Qwen/Qwen3-4B --max_model_len 8192
```



## Arguments

### Required
- `--model_name <model>`: HuggingFace model name or path
- `--max_model_len <length>`: Maximum sequence length (e.g., 8192, 32768)

### Optional
- `--limit <number>`: Number of samples per task (default: all)
- `--batch_size <size>`: Batch size (default: auto)
- `--gpu_memory_util <ratio>`: GPU memory utilization 0.0-1.0 (default: 0.8)
- `--log_samples`: Enable sample logging for debugging
- `--help`: Show help message

## Testing Models Examples
- Standard Models
  - `meta-llama/Llama-3.2-3B-Instruct`
  - `meta-llama/Llama-3.1-8B-Instruct`

- Thinking Models (enable_thinking=True,think_end_token='</think>')
  - `Qwen/Qwen3-1.7B`
  - `Qwen/Qwen3-4B`
  - `Qwen/Qwen3-8B`
  - `Qwen/Qwen3-14B`

## Task Groups Evaluated

1. **Inpatient Tasks**: Patient management and operational predictions
2. **Measurement Tasks**: 
   - Lab values prediction
   - Vital signs prediction
3. **Diagnosis Tasks**:
   - New diagnosis prediction
   - Recurrent diagnosis prediction

## Requirements

- SLURM cluster access (for SLURM script)
- Conda environment: `bids_lm_eval` with vllm 0.9.1
- HuggingFace authentication for gated models

## Output Structure

```
results/ehr_llm/ehrshot/
├── task_inpatient/max_len_8192/
├── task_measurement/
│   ├── labs/max_len_8192/
│   └── vitals/max_len_8192/
└── task_diagnosis/
    ├── new/max_len_8192/
    └── recurrent/max_len_8192/
```

## Common Usage Patterns

### Debugging commands

```bash

lm_eval \
   --model vllm \
   --model_args "pretrained=openai/gpt-oss-20b,tensor_parallel_size=1,data_parallel_size=1,dtype=bfloat16,max_model_len=8192,gpu_memory_utilization=0.7,enable_thinking=True'" \
   --apply_chat_template \
   --include_path bids_tasks/ehr_llm \
   --tasks simple_dev_task \
   --batch_size auto \
   --output_path results/debug \
   --log_samples \
   --limit 10


lm_eval \
   --model vllm \
   --model_args "pretrained=Qwen/Qwen3-4B,tensor_parallel_size=1,data_parallel_size=1,dtype=bfloat16,max_model_len=8192,gpu_memory_utilization=0.7,enable_thinking=True" \
   --apply_chat_template \
   --include_path bids_tasks/ehr_llm \
   --tasks simple_dev_task \
   --batch_size auto \
   --output_path results/debug \
   --log_samples \
   --limit 10


lm_eval --model hf \
    --model_args pretrained=openai/gpt-oss-20b \
    --apply_chat_template \
    --include_path bids_tasks/ehr_llm \
    --tasks simple_dev_task \
    --batch_size auto \
    --output_path results/debug \
    --log_samples \
    --limit 10


lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen3-4B,dp_size=1,tp_size=1,dtype=auto \
    --apply_chat_template \
    --include_path bids_tasks/ehr_llm \
    --tasks simple_dev_task \
    --batch_size auto
```


### Development & Testing
```bash
# Quick test (10 samples)
sbatch lm_eval_ehrllm_ehrshot_task_slurm.sh --model_name Qwen/Qwen3-1.7B --max_model_len 8192 --limit 10 --log_samples

# Medium test (100 samples)
sbatch lm_eval_ehrllm_ehrshot_task_slurm.sh --model_name meta-llama/Llama-3.2-3B-Instruct --max_model_len 8192 --limit 100
```

### Full Runs
```bash
# Full evaluation
sbatch lm_eval_ehrllm_ehrshot_task_slurm.sh --model_name meta-llama/Llama-3.1-8B-Instruct --max_model_len 8192

# Large context evaluation
sbatch lm_eval_ehrllm_ehrshot_task_slurm.sh --model_name Qwen/Qwen3-4B --max_model_len 32768
```

## Troubleshooting

- **CUDA OOM**: Reduce `--gpu_memory_util` to 0.6 or lower
- **Gated models**: Run `huggingface-cli login` first
- **Context length**: Use higher `--max_model_len` for EHR data
- **Debugging**: Add `--log_samples --limit 10` for detailed output

## Features

- **Thinking Model Support**: Automatic reasoning chain support for compatible models
- **Memory Optimization**: Configurable GPU memory and batch size settings
- **Error Handling**: Comprehensive validation and progress tracking
- **Cache Management**: Automatic redirection to scratch space
- **SLURM Integration**: Optimized for H100/H200 GPUs with proper resource allocation 