# EconCausal: A Context-Aware Causal Reasoning Benchmark for LLMs in Social Science

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset Scale](https://img.shields.io/badge/Data_Scale-10.4K_Tuples-green)](./data)
[![Paper](https://img.shields.io/badge/KDD_2026-Benchmark_Track-red)](https://kdd.org/kdd2026)

**EconCausal** is a high-fidelity benchmark designed to evaluate the **context-aware causal reasoning** capabilities of Large Language Models (LLMs) in economics and social science.

Unlike the natural sciences—where causal laws often aim for universality—economic causal claims are fundamentally **context-dependent**. The same intervention may yield different causal effects depending on institutional settings, time periods, enforcement regimes, or market structures.

EconCausal introduces a novel **4-tuple causal representation**  
\[
(T, O, D, C)
\]
to rigorously test whether LLMs can integrate **contextual constraints** into causal judgment rather than relying on surface-level correlations.

---

## 🌟 Key Features

- **Context-Aware Causal Framework**  
  Moves beyond binary causal pairs to a structured **4-tuple** formulation:
  - **T**: Treatment  
  - **O**: Outcome  
  - **D**: Direction of causal effect (`+`, `-`, `None`, `Mixed`)  
  - **C**: Socio-economic context (institutional, temporal, geographic)

- **Empirical Rigor**  
  Constructed from **4,176 peer-reviewed empirical studies**, primarily drawn from:
  - NBER Working Papers later published in top economics and finance journals
  - Venues including *AER, QJE, JPE, Review of Economic Studies, Econometrica, JF, JFE, RFS*  
  All causal claims are supported by explicit identification strategies (e.g., DiD, IV, RDD, RCT).

- **Large-Scale & High-Quality Dataset**  
  - **10,439 validated causal 4-tuples**
  - Coverage across **20+ JEL categories**
  - Filtered through a conservative multi-stage pipeline combining LLM consensus and expert-aligned quality control

- **Challenging Reasoning Tasks**  
  Designed to probe deeper causal reasoning beyond memorization:
  1. **Causal Sign Identification**
  2. **Context-Dependent Sign Prediction**
  3. **Misinformation-Robust (Counter-Context) Reasoning**

---

## 🧠 Benchmark Tasks

### Task 1: Causal Sign Identification
Given a context **C** and a treatment–outcome pair **(T, O)**, predict the causal direction **D** under that context.

### Task 2: Context-Dependent Sign Prediction
Given an observed causal relationship under one context, infer how the **same or comparable** treatment–outcome pair behaves under a **different context**.

### Task 3: Misinformation-Robust (Counter-Context) Reasoning
Extend Task 2 by providing an **incorrect or misleading example sign**, testing whether models can discount spurious evidence and reason based on contextual grounding.

---

## 📂 Repository Structure

```text
.
├── data/
│   ├── step1_raw_extractions.csv      # 17.6k initial LLM-extracted causal candidates
│   ├── step2_paper_metadata.csv       # 4.1k paper-level context & identification methods
│   ├── step3_matched_triplets.csv     # 13.4k context-matched causal triplets
│   └── step4_final_benchmark.csv      # 10.4k QC-passed causal 4-tuples (Final Benchmark)
│
├── prompts/                           # Transparency of generative AI usage
│   ├── extraction_prompts.md          # Prompts used in Steps 1–3
│   └── qc_critic_criteria.md          # Step 4: Multi-critic quality control rubric
│
├── scripts/
│   ├── evaluate_baselines.py          # Reproduction script for KDD paper results
│   └── data_loader.py                 # Utilities for loading EconCausal 4-tuples
│
├── CITATION.cff                       # Citation metadata
└── README.md
```

---

## 📁 Dataset Description

The final benchmark dataset (`data/step4_final_benchmark.csv`) contains **10,439 causal 4-tuples** extracted from peer-reviewed empirical research in economics and finance.

Each entry is represented as a structured tuple:
- **T (Treatment)**: the intervention or policy variable  
- **O (Outcome)**: the affected economic or social outcome  
- **D (Direction)**: causal sign (`+`, `-`, `None`, `mixed`) based on the authors’ preferred specification  
- **C (Context)**: study-specific institutional, temporal, and geographic conditions  

All tuples are grounded in papers published in top-tier journals and supported by explicit econometric identification strategies.  
Original paper PDFs are not redistributed due to copyright restrictions; instead, sufficient metadata is provided to enable independent verification.

---

## 🔁 Reproducibility

The benchmark results reported in the accompanying KDD paper can be reproduced using the provided evaluation scripts.

### Installation
```bash
pip install -r requirements.txt
```
### Run Evaluation
```bash
python scripts/evaluate_baselines.py
```
The evaluation pipeline uses fixed dataset splits and controlled randomness to ensure reproducibility of all reported results.

---

## ⚖️ Ethical Considerations
- All data are derived from publicly available, peer-reviewed academic research.
- No new human-subject data were collected for this benchmark.
- No personally identifiable information (PII) is included.
- Copyrighted texts (e.g., paper PDFs) are not redistributed.
- Generative AI models were used for extraction, refinement, and quality control, and their usage is fully documented via prompts and criteria provided in this repository.

---

## 📜 License
- Dataset: Creative Commons Attribution 4.0 (CC BY 4.0)
- Code: Released for research and academic use

Please refer to the LICENSE file for the full license terms.

---

## 📝 Citation
If you use EconCausal in your research, please cite the following paper:
```bibtex
@inproceedings{lee2026econcausal,
  title     = {EconCausal: A Context-Aware Causal Reasoning Benchmark for LLMs in Social Science},
  author    = {Lee, Donggyu and Yun, Hyeok and Cha, Meeyoung and Park, Sungwon and Park, Sangyoon and Kim, Jihee},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2026}
}
```
Citation metadata is also provided in CITATION.cff.

---

## 📬 Contact
For questions, issues, or suggestions regarding the dataset or benchmark,
please open a GitHub issue or contact the authors listed in the paper.
