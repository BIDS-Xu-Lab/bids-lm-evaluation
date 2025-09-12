# This is the custom benchmark framework frmo the lm-evaluation-harness for our labs LLM-related projects.

## Installation
- Follow the installation instructions in the lm-evaluation-harness README.md
    ```bash
    conda create --name bids_lm_eval python=3.12
    conda activate bids_lm_eval
    git clone --depth 1 git@github.com:BIDS-Xu-Lab/bids-lm-evaluation.git
    cd bids-lm-evaluation
    pip install -e .
    ```
- pip install lm_eval[vllm] vllm==0.9.1 # Otherwise [vllm: Sampled token IDs exceed the max model length](https://github.com/EleutherAI/lm-evaluation-harness/issues/3134)
- pip install "transformers<4.54.0"  # See: [vllm-ascend issue #2046: All vLLM <= v0.10.0 and transformers>=4.54.0 will encounter this issue](https://github.com/vllm-project/vllm-ascend/issues/2046)

## Usage
### Exsitng tasks
 - To use existing tasks, check out lm_eval/tasks folder. 
- 
### Add new tasks
 - Add new tasks to the `bids_tasks` folder and make sure to add `include_path` for this folder in testing lm_eval command. 
