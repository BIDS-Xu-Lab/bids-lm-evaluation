# This is the custom benchmark framework frmo the lm-evaluation-harness for our labs LLM-related projects.

## Installation
- Follow the installation instructions for compatibility 
    ```bash
    conda create --name bids_lm_eval python=3.12
    conda activate bids_lm_eval
    git clone --depth 1 git@github.com:BIDS-Xu-Lab/bids-lm-evaluation.git
    cd bids-lm-evaluation
    pip install uv
    uv pip install -e .
    uv pip install "lm_eval[vllm]" "vllm==0.8.5" # for MOE model compatibility, do not support gpt-oss quantization

    ```


## Usage
 - To use existing tasks, check out lm_eval/tasks folder. 
 - Add new tasks to the `bids_tasks` folder and make sure to add `include_path` for this folder in testing lm_eval command. 
 - check out the jobs_run folder to find the scripts run on specific YCRC HPC
