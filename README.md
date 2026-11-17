# [Project Title] — LLaMEA-BO Assignment



## 1. Overview

| Parameter | Value / Details |
| :--- | :--- |
| **Group ID ($g$)** | `12` |
| **Assigned Random Seeds ($s_{g,r} = 10g + r$, $r = 0, \dots, 9$)** | `[120, 121, 122, 123, 124, 125, 126, 127, 128, 129]` |
| **Dimensionality** | $d = 6$ |
| **Selected BBOB Function (Instance 1)** | `f15: Rastrigin` |
| **Selected Real-World Problem** |Real world problem **1124 - Frequency Modulated Sound Waves** - [Practical Problem](https://github.com/BayesOptApp/Practical_Problems/blob/main/Practical_Problems/Problems/Frequency_Modulated_Sound_Waves.m)  |
| **Objective Convention** | All problems treated as minimization tasks; however, during llamea evolution loop the feedback score is the **AOCC** which we want to maximize.  |


---



## 2. Environment Setup & Installation

* **Operating System & Hardware:** 
<!-- [Document the machines used for running experiments, including CPU/GPU/TPU specs, RAM, and execution time notes]. -->

CPU: Intel Core i7-14700, 20 cores / 28 threads, up to 5.4 GHz.

GPU: NVIDIA GeForce RTX 4060, 8 GB VRAM, CUDA 13.2.

Execution time: 5-7 hours for LLaMEA loop, 5-7 hours for final evaluation (the largest part of this are the BO-based algorithms).

* **CPU & Hardware Support:** [Explain how the codebase runs on CPUs and document any GPU/TPU-specific dependencies or flags].

LLaMEA loop (stage1) and final algorithm evaluation (stage2) run on the CPU. 
The local LLM is served on the GPU. Running the LLM on CPU is possible; however, sufficient memory is necessary (recommended 16GB+). 
This is variable dependent on local LLM size. If using free API-tiers GPU considerations can be skipped. Otherwise, ensure to consider sufficient VRAM + CPU RAM memory to fit local model. 
This experiment requires ~10GB for a quantized LLM; more is better. 

* **Python & Dependencies:**
```bash
# Python version: 
python==3.11
```

```bash
botorch==0.18.1
ioh==0.3.22
llamea==1.4.2
cma==4.5.0
```

* **Installation Steps:**

Package manager. 
```bash
# Recommended python package manager (UV)
# FOr linux/macos
curl -LsSf https://astral.sh/uv/install.sh | sh

# For windows PS
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Environment setup (linux). 
```bash
# Navigate to the root of the code repository
# init venv
uv venv .venv --python python3.11

# activate venv
source .venv/bin/activate

# install pip requirements
uv pip install -r requirements.txt
```



---



## 3. LLM Provider Configuration

* **Selected Free-to-Use LLM & Provider:** [Insert model name and free API/framework used; note that only free-to-use LLMs are permitted].

NOTE: recommended to have access to GPU (8GB+ VRAM), otherwise free-API tiers with high Requests Per Day limits and token windows are recommended. 
This setup uses local `*.GGUF` models and serves these locally with **ollama**. 

    - Chosen model: `Qwen/Qwen2.5-Coder-14B-Instruct-GGUF` 
    - Quantization: Q4_K_M (K-Quant + medium setting)
    - ~8.99 GB memory uptake
    - Can be downloaded from [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF/blob/main/qwen2.5-coder-14b-instruct-q4_k_m.gguf)

* **Configuration Steps:** [Explain how to set API keys, environment variables, endpoints, model parameters, etc.].

Two methods are covered: 1) limited access on a remote machine; requires scp files to remote machine from local machine. 2) working only locally, no need for scp. 

Ollama setup.

```bash
# Linux (install ollama binary on machine)
wget https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst

# OPTIONAL: if installed it locally and need to scp to remote:
scp your_downloaded_ollama.tar.zst username@domain:/path_to_your_remote

# Extract ollama 
tar -I zstd -xf ollama-linux-amd64.tar.zst

# OPTIONAL: Set env variables (IMPORTANT: this depends on your machine 
# and file system; this is specifically for a remote workflow)
echo 'export OLLAMA_HOST="0.0.0.0:11434"' >> ~/.bashrc
echo 'export OLLAMA_MODELS="/local/user/ollama/ollama_data"' >> ~/.bashrc
source ~/.bashrc
```

GGUF model installation. 

Method 1) limited access with Remote GPU machine; need to scp files to remote. 
```bash
# Follow the steps on your local machine (should take 15-30 minutes)
curl -LsSf https://llama.app/install.sh | sh

# downloads two shards locally
llama cli -hf Qwen/Qwen2.5-Coder-14B-Instruct-GGUF --model qwen2.5-coder-14b-instruct-q4_k_m.gguf

# merge the two shards
llama-gguf-split --merge path_to_shard_1 

# scp the merged model to remote
scp your_downloaded_model.GGUF username@domain:/path_to_your_remote

# create the model 
echo "FROM ./Qwen2.5-Coder-14B-Q4_K_M.gguf" > Modelfile
ollama create choose_model_name -f Modelfile

# serve model on remote
ollama serve &
```

Method 2) everything locally.
```bash
# pull model from HF
ollama run hf.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF:Q4_K_M 

# serve model locally
ollama serve &
```

The LLM is configured in the `config/config.py` file. If using a different GGUF model, simply change the name
of the model:
```bash 
# check the model name 
curl "http://localhost:11434/v1/models" 
```
The `temperature` and `top_p` parameters can be modified according to desired settings. Other options can be added to the configuration file. 
Ollama listens by default on "http://localhost:11434/v1/". This default url is set in the config file as well. Change it to a different URL if necessary. 



---



## 4. Configuring Prompts & Hyperparameters

* **LLaMEA Outer-Loop & Evolutionary Settings:** [Document population size, number of parents, prompt templates, and any creative modifications made to LLaMEA and why].

**Evolutionary templates**
These parameters were find through trial and error. (1,1) and (1+1) did not work sufficiently well, increasing the number of parents and offspring is necessary to generate functional python code (for our local model). In Li et al. an ablation study on these two parameters was done, (8+16) worked best, with diminished gains from other settings. (8+16) was too slow during our trials, therefore a (2+5) approach was chosen. 
Population size: 5
Number of parents: 2 
Elitism: True

**Prompts**
A single role prompt is used throughout our experiments: 
```python 
ENGINEERED_ROLE_PROMPT = """You are an AI that generates Python optimization code by using Bayesian Optimization.
"""
```

Two prompt task prompt strategies were investigated and iterated over. Both prompts focus on explicit rules, pythonic structure, and objective function optimization direction (also stating the objective of maximizing AOCC; thus, achieving early success is best). 
Both prompts use the same Sobol sampling as initialization strategy. 

All experiment prompts can be found in `stage1/prompts/exp_prompts.py`. Other prompts which were trialled are in `stage1/prompts/prompts.py`. 

`BO_
1. Explicit BO skeleton with functional BO code. Qwen was very prone to producing import errors and shape mismatches. Use an explicit functional code skeleton and instead focus on small changes helped Qwen produce valid code. 

2. Iterated BO prompt where encountered errors were added as explicit rules to circumvent recurrence. Only the desired class constructor structure and __call__ structure were outlined. Then, a bulk of explicit rules to account for Qwen's many pythonic errors. 
Finally, a small snippet to remind the model of feedback importance during the EA loop. The hopes with this prompt is that due to the lack of an explicit BO skeleton, the model can generate more creative solutions. 

**LLaMEA** modifications. 

No modifications were made to the LLaMEA library. Instead, a lot of the main evaluator and experiment logging was taken from the [LLaMEA github examples](https://github.com/XAI-liacs/LLaMEA/blob/main/examples/black-box-optimization.py). 


* **Stage 1 Budgets & Selection:**

  * Max candidate algorithms per LLaMEA search: `100`

  * Inner-loop budget per run: `10d = 60` evaluations (including initialization)

  * Independent runs per candidate: `10` (using group seeds)

  * Selection criterion: Area Over the Convergence Curve (AOCC) aggregated across the 10 runs

* **Stage 2 Budgets & Baselines:**

  * Evaluation budget per run: `100d = 600` evaluations (including initialization)

  * Independent runs: `10` (using group seeds)

  * Baselines compared: Random Search, CMA-ES and BoTorch BO (document surrogate model, acquisition function, and initialization strategies for all methods)
        - Initialization strategy: for all methods (except for Random Search), use the same Sobol sampling as developed by Qwen in Llamea. 
        - Surrogate model: All three Bayesion-optimization methods use a GP surrogate model with Exact Marginal Log Likelihood.
        - Acquisition function: Both LLaMEA generated models use qLogExpectedImprovement. Our BOTorch BO baseline uses LogExpectedImprovement. Difference is sequential vs batch processing of candidates. 
    

---



## 5. Running Experiments (Stage 1 & Stage 2)

* **Reproducibility & Randomness:** 

[Detail how seeds are propagated across LLaMEA, inner optimization loops, baselines, and libraries, plus any uncontrollable sources of randomness].

Each generated algorithm by Qwen in the Llamea loop is passed inside our Evaluator class (in `stage1/evaluator.py`). This class handles evaluation of the generated candidate (and necessary fallbacks in case of invalid code) across our 10 seeds. We use the AOCC logging classes and functions inside `stage1/utils.py` to compute the mean and std aggregated across the 10 seeds. Both the array of scores, and mean and std values are added to metadata. Random seeds are set inside the inner loop before each evaluation on the problem function. It is then passed as a parameter to the algorithm class, which uses it to control randomness in initial design.
For the baselines, the seeds are used for initial design as well, as well as random fallback sampling in botorch bo baseline. The libraries (BoTorch, cma) use the initialized np and torch random seeds for random number generation. 

TODO: improve randomness explanation. Elaborate further. 

* **Stage 1 — Algorithm Generation:**

For all commands, ensure you are in the root of the repository. 

```bash

# To run the algorithm generation simply use the following command from the root of the repository:
python -m stage1.bo_gen # runs for both problems by default. 

```

* **Stage 2 — Generalization Evaluation & Baselines:**
For all commands, ensure you are in the root of the repository. 

First copy the generated algorithms over to the `algorithms/` folder and select the prompt sub-folder to which it belongs. 
```bash
# to run the evaluation of all algorithms
python -m stage2.runner
```

* **Generating Logs & Convergence Plots:**

```bash
# Plot llamea convergence over AOCC
python -m plots.plot_llamea 

# Plot final evaluation of stage 2
python -m plots.plot_convergence
```


---



## 6. Generated Algorithms: Workflow & Pseudocode



### 6.1 Algorithm Generated on BBOB Function

* **Component Analysis:** [Describe surrogate model, acquisition function, initialization, parameter selection/adaptation, and how it differs from conventional BO].

* **Pseudocode / Workflow:**

  ```text

  [Insert step-by-step pseudocode or workflow for the algorithm generated on the BBOB target]

  ```



### 6.2 Algorithm Generated on Real-World Problem

* **Component Analysis:** [Describe surrogate model, acquisition function, initialization, parameter selection/adaptation, and how it differs from conventional BO].

* **Pseudocode / Workflow:**

  ```text

  [Insert step-by-step pseudocode or workflow for the algorithm generated on the real-world target]

  ```



---



## 7. Results & Attribution

* **Outputs & Logs Location:** 
- LLamea generation logs are saved under `llamea_logs/BO_STUB_TASK_PROMPT_1`.
- Stage 2 evaluation results are saved under `results/stage2/first_results`
- Stage 1 & 2 plots are saved under `plots/output/`. `llamea_aocc.pdf` includes the convergence during algorithm generation. `convergence_bbob.pdf` shows the convergence of all algorithms on `f15` as incumbent error against evaluations. `convergence_real` shows the convergence of all algorithms on the practical problem 1124, again, as incumbent error. 


* **External Code & Citations:** [Add inline code comments and list any reused code/repositories with links and original authorship to avoid plagiarism, plus cite relevant references].

- [Practical problems](https://github.com/BayesOptApp/Practical_Problems/blob/main/Practical_Problems/Problems/Frequency_Modulated_Sound_Waves.m)
- [AOCC logging](https://github.com/XAI-liacs/LLaMEA/llamea/utils.py)
- [Evaluator skeleton](https://github.com/XAI-liacs/LLaMEA/examples)

- Li, W., van Stein, N., Bäck, T., & Raponi, E. (2026, July). Llamea-bo: A large language model evolutionary algorithm for automatically generating bayesian optimization algorithms. In Proceedings of the Genetic and Evolutionary Computation Conference (pp. 909-918).