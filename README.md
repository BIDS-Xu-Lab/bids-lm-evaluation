# This is the custom benchmark framework frmo the lm-evaluation-harness for our labs LLM-related projects.

## Installation
- Follow the installation instructions in the lm-evaluation-harness README.md
- [Use the guide from gpt-oss for compatible vllm versio](https://huggingface.co/openai/gpt-oss-20b)
    ```bash
    conda create --name bids_lm_eval python=3.12
    conda activate bids_lm_eval
    git clone --depth 1 git@github.com:BIDS-Xu-Lab/bids-lm-evaluation.git
    cd bids-lm-evaluation
    pip install -e .
    pip install uv
    uv pip install --pre vllm==0.10.1+gptoss \
      --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
      --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
      --index-strategy unsafe-best-match
    ```


## Usage
 - To use existing tasks, check out lm_eval/tasks folder. 
 - Add new tasks to the `bids_tasks` folder and make sure to add `include_path` for this folder in testing lm_eval command. 
 - check out the jobs_run folder to find the scripts run on specific YCRC HPC
