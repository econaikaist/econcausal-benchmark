# EconCausal: A Context-Aware Causal Reasoning Benchmark for LLMs in Social Science

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset Scale](https://img.shields.io/badge/Data_Scale-10.4K_Tuples-green)](./data)
[![Paper](https://img.shields.io/badge/KDD_2026-Benchmark_Track-red)](https://kdd.org/kdd2026)

**EconCausal** is a high-fidelity benchmark designed to evaluate the **context-aware causal reasoning** capabilities of Large Language Models (LLMs) in the domain of economics and social science.

Unlike the natural sciences, where causal laws often aim for universality, economic causal claims are fundamentally **context-dependent**. EconCausal introduces a novel **4-tuple framework $(T, O, D, C)$** to assess whether models can integrate institutional, temporal, and geographic constraints into their causal judgment.

---

## 🌟 Key Features

- **Context-Aware Framework**: Moves beyond simple binary causal pairs to a nuanced **4-tuple** structure: $(T, O, D, C)$ — Treatment, Outcome, Direction, and Context.
- **Empirical Rigor**: Built from **4,176 peer-reviewed papers** (NBER, JPE, AER, etc.), ensuring all claims are grounded in rigorous econometric identification strategies (e.g., DiD, IV, RDD).
- **Large-Scale & High-Quality**: Features **10,439 validated causal tuples** across 20+ JEL categories, refined through a multi-stage expert-verified pipeline.
- **Novel Reasoning Tasks**: Includes three tasks: **Existence**, **Direction**, and **Counter-Context**, challenging models to predict how causal signs flip under different institutional settings.

---

## 📂 Repository Structure

```text
.
├── data/
│   ├── step1_raw_extractions.csv      # 17.6k initial extractions
│   ├── step2_paper_metadata.csv       # 4.1k paper-level context & methods
│   ├── step3_matched_triplets.csv     # 13.4k context-matched triplets
│   └── step4_final_benchmark.csv      # 10.4k QC-passed 4-tuples (Final Benchmark)
├── prompts/                           # Transparency of Generative AI usage
│   ├── extraction_prompts.md          # Step 1-3 prompts
│   └── qc_critic_criteria.md          # Step 4: The 5 strict QC criteria
├── scripts/
│   ├── evaluate_baselines.py          # Reproduction script for paper results
│   └── data_loader.py                 # Utility to load EconCausal 4-tuples
├── CITATION.cff                       # BibTeX information
└── README.md

@inproceedings{lee2026econcausal,
  title={EconCausal: A Context-Aware Causal Reasoning Benchmark for LLMs in Social Science},
  author={Lee, Donggyu and others},
  booktitle={KDD},
  year={2026}
}
