# EHRShot Evaluation Scripts

> **Bouchet Cluster Only**  
> These scripts are designed to run on the **Bouchet cluster** specifically.  
> EHRShot data is located at: `/home/yl2342/project_pi_hx235/yl2342/data/ehrshot/visit_oriented_ehr/`

This directory contains scripts for evaluating language models on the EHRShot benchmark using the lm-evaluation-harness framework.

## Scripts

- **`lm_eval_ehrllm_ehrshot_task_slurm.sh`**: SLURM-based evaluation script for cluster submission
- **`lm_eval_ehrllm_ehrshot_task.sh`**: Interactive evaluation script for direct execution on compute nodes

Both scripts are fully aligned and use the same task configurations, conda environment (`bids_lm_eval`), and support thinking models with automatic reasoning chain capabilities.

## Quick Start

### SLURM Script (Recommended)

```bash
# Basic usage
sbatch lm_eval_ehrllm_ehrshot_task_slurm.sh --model_name meta-llama/Llama-3.2-3B-Instruct --max_model_len 8192

# With thinking models
sbatch lm_eval_ehrllm_ehrshot_task_slurm.sh --model_name Qwen/Qwen3-1.7B --max_model_len 8192

# Testing with limited samples
sbatch lm_eval_ehrllm_ehrshot_task_slurm.sh --model_name meta-llama/Llama-3.1-8B-Instruct --max_model_len 8192 --limit 100
```

### Interactive Script

```bash
# Basic usage (run on compute node)
./lm_eval_ehrllm_ehrshot_task.sh --model_name meta-llama/Llama-3.2-3B-Instruct --max_model_len 8192

# With debugging
./lm_eval_ehrllm_ehrshot_task.sh --model_name Qwen/Qwen3-1.7B --max_model_len 8192 --limit 10 --log_samples
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

The scripts evaluate the following EHRShot task groups:

1. **Inpatient Tasks** (`group_ehrshot_inpatient_tasks_gu`): Patient management and operational predictions
2. **Measurement Tasks**: 
   - **Lab Tasks** (`group_ehrshot_measurement_lab_tasks_gu`): Lab values prediction
   - **Vital Tasks** (`group_ehrshot_measurement_vital_tasks_gu`): Vital signs prediction
3. **Diagnosis Tasks**:
   - **New Diagnosis** (`group_ehrshot_new_diagnosis_tasks_gu`): New diagnosis prediction
   - **Recurrent Diagnosis** (`group_ehrshot_recurrent_diagnosis_tasks_gu`): Recurrent diagnosis prediction

## SLURM Configuration

The SLURM script is configured with:
- **Partition**: `gpu`
- **GPUs**: `h100:2` (2x H100 GPUs)
- **Time Limit**: 24 hours
- **Memory**: 512GB
- **CPUs**: 16 cores
- **Nodes**: 1

## Requirements

- SLURM cluster access (for SLURM script)
- Conda environment: `bids_lm_eval` with vllm 0.9.1
- HuggingFace authentication for gated models
- Access to Bouchet cluster resources

## Environment Configuration

The scripts automatically configure:
- **HuggingFace Cache**: Uses scratch space (configured in ~/.bashrc)
- **PyTorch Settings**: `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`
- **vLLM Optimization**: Usage stats disabled, tensor/data parallelism configured
- **Torch Compile Cache**: Automatic cache directory management

## Output Structure

Results are saved to `/gpfs/radev/home/yl2342/project/bids-lm-evaluation/results/`:

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
- **Gated models**: Run `huggingface-cli login` first or ensure HF token is in `$HF_HOME/token`
- **Context length**: Use higher `--max_model_len` for EHR data
- **Debugging**: Add `--log_samples --limit 10` for detailed output
- **Disk quota**: Cache directories are automatically configured to use scratch space

## Features

- **Thinking Model Support**: Automatic reasoning chain support for compatible models with `enable_thinking=True` and `think_end_token='</think>'`
- **Memory Optimization**: Configurable GPU memory and batch size settings with data/tensor parallelism
- **Error Handling**: Comprehensive validation, progress tracking, and exit code handling
- **Cache Management**: Automatic redirection to scratch space for HuggingFace and PyTorch caches
- **SLURM Integration**: Optimized for H100 GPUs with proper resource allocation and logging
- **Environment Cleanup**: Automatic conda deactivation and module unloading 