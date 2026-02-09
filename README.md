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
  - **D**: Direction of causal effect (`+`, `-`, `None`, `mixed`)  
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
