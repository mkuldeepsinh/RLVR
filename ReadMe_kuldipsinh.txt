Project Name: Improving Mathematical Reasoning in LLMs: RLVR to DAPO

Group Name and Contact Details:
1) Kuldipsinh (Team Leader) | Roll No: U23AI023 
2) Priya  |  U23AI028
3) Jaimini | U23AI066

Course:
LAB VI — Reinforcement Learning with Verifiable Rewards

Project Summary:
This project reproduces the One-Shot RLVR baseline and extends it with DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) for mathematical reasoning on Qwen2.5-Math-1.5B. The workflow includes RLVR baseline benchmarking, DAPO-style GRPO fine-tuning with QLoRA, adapter merging, and presentation/report generation.

Repository Contents:
- RLVR.ipynb: RLVR baseline setup and benchmark execution workflow
- DeePO_DAPO.ipynb: DAPO implementation and fine-tuning pipeline
- results_from_colab/depo model/: saved LoRA adapter artifacts from Colab
- results_from_colab/merge model/: merged model artifacts/config for inference
- DAPO_Implementation_Report.md: technical report of methodology and results
- generate_ppt.py: original project presentation generator
- generate_ppt_enhanced.py: improved presentation generator

Environment Setup Options:

Option A: Conda environment (local)
1. Install Anaconda/Miniconda.
2. Open terminal in project folder.
3. Create environment:
   conda create -n rlvr_dapo python=3.10 -y
4. Activate environment:
   conda activate rlvr_dapo
5. Install packages:
   pip install -U pip
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install transformers datasets accelerate peft trl bitsandbytes matplotlib numpy python-pptx jupyter
6. Launch notebook server:
   jupyter notebook

Option B: Pip + venv (local)
1. Open terminal in project folder.
2. Create virtual environment:
   python3 -m venv .venv
3. Activate environment:
   source .venv/bin/activate
4. Install packages:
   python3 -m pip install -U pip
   python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   python3 -m pip install transformers datasets accelerate peft trl bitsandbytes matplotlib numpy python-pptx jupyter

Option C: Google Colab (recommended for this project)
1. Open RLVR.ipynb or DeePO_DAPO.ipynb in Colab.
2. Set runtime type to GPU (Tesla T4).
3. Run dependency installation cells in the notebook.
4. Run all cells in sequence.

Hardware Note:
Project was primarily developed and tested on Google Colab Tesla T4 (~15 GB VRAM) with QLoRA (4-bit NF4).

Important:
Before final submission, replace placeholder roll numbers/mobile numbers above with actual group details.
