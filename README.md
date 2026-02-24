<h1 align="center">EconCausal: A Context-Aware Causal Reasoning Benchmark for Large Language Models</h1>

<p align="center">
  <b>Donggyu Lee<sup>1</sup>, Hyeok Yun<sup>2</sup>, Meeyoung Cha<sup>3</sup>, Sungwon Park<sup>4*</sup>, Sangyoon Park<sup>5</sup>, Jihee Kim<sup>2*</sup></b>
</p>

<p align="center">
  <sup>1</sup>Graduate School of Data Science, KAIST &nbsp;
  <sup>2</sup>College of Business, KAIST &nbsp;
  <sup>3</sup>MPI-SP, Germany<br>
  <sup>4</sup>School of Computing, KAIST &nbsp;
  <sup>5</sup>Division of Social Science, HKUST
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2510.07231"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg" alt="arXiv"></a>
  <a href="https://github.com/econaikaist/econcausal-benchmark"><img src="https://img.shields.io/badge/GitHub-Repository-181717.svg" alt="GitHub"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg" alt="License"></a>
</p>

---

## Overview

Socio-economic causal effects depend heavily on their specific institutional and environmental context. A single intervention can produce opposite results depending on regulatory or market factors. This poses a significant challenge for LLMs in decision-support roles: **can they distinguish structural causal mechanisms from surface-level correlations when the context changes?**

**EconCausal** is a large-scale benchmark comprising **10,490 context-annotated causal triplets** extracted from **2,595 high-quality empirical studies** published in top-tier economics and finance journals. Through a rigorous four-stage pipeline combining multi-run consensus, context refinement, and multi-critic filtering, each claim is grounded in peer-reviewed research with explicit identification strategies.

<p align="center">
  <img src="figures/intro.png" width="85%" alt="EconCausal Overview">
</p>

### Key Findings

- Top models achieve ~88% accuracy in fixed, explicit contexts
- Performance **drops 32.6 pp** under context shifts (Task 2)
- Performance **collapses to 37%** when misinformation is introduced (Task 3)
- Models achieve only **9.5% accuracy** on null effects, exposing a fundamental gap between pattern matching and genuine causal reasoning

---

## Dataset

### Data Statistics

| Statistic | Value |
|---|---|
| Total causal triplets | 10,490 |
| Source papers | 2,595 |
| Publication years | 1991 -- 2025 |
| Economics journals | 5 (AER, QJE, JPE, ReStud, ECMA) |
| Finance journals | 3 (JFE, JF, RFS) |
| Domain split | Economics 67.7% / Finance 32.3% |

### Data Format

Each causal triplet includes:

| Field | Description |
|---|---|
| `treatment` | Independent variable / intervention |
| `outcome` | Dependent variable / affected endpoint |
| `sign` | Direction of causal effect (`+`, `-`, `None`, `mixed`) |
| `context` | Institutional and environmental context (max 100 words) |
| `identification_methods` | Identification strategies (DiD, IV, RCT, RDD, etc.) |
| Paper metadata | `paper_id`, `title`, `author`, `publication_year`, `published_venue`, `jel_codes`, `paper_url` |

---

## Benchmark Tasks

EconCausal includes three progressively challenging evaluation tasks probing context-dependent causal reasoning.

### Task 1: Causal Sign Prediction (947 econ + 860 finance)

Given a context and a treatment-outcome pair, predict the causal sign. Tests whether LLMs can internalize economic causalities from peer-reviewed research.

### Task 2: Context-Dependent Sign Prediction (284 instances)

Given a known causal effect under context c1, predict the sign of the same treatment-outcome pair under a different context c2. Tests whether LLMs understand that causality is context-dependent.

### Task 3: Misinformation-Robust Sign Prediction (852 instances)

Same as Task 2, but with deliberately incorrect sign information. Tests whether LLMs can discount misinformation and perform robust, context-grounded reasoning.

---

## Repository Structure

```
econcausal-benchmark/
├── data/
│   ├── causal_triplets/        # 10,490 causal triplets (csv + jsonl)
│   ├── tasks/                  # Benchmark tasks (csv + jsonl)
│   │   ├── task1_econ.*        # Task 1 - Economics
│   │   ├── task1_finance.*     # Task 1 - Finance
│   │   ├── task2.*             # Task 2
│   │   └── task3.*             # Task 3
│   └── metadata/               # NBER paper metadata
├── prompts/
│   └── evaluation/             # Benchmark task prompts (Tasks 1-3)
├── scripts/
│   ├── llm_evaluation/
│   └── common/
├── figures/
├── LICENSE
└── README.md
```

---

## Getting Started

### Requirements

```bash
pip install openai pandas openpyxl tqdm numpy
```

### Running LLM Evaluation

```bash
cd scripts/llm_evaluation

# Run all tasks with default models
python run_evaluation.py

# Run specific tasks
python run_evaluation.py --tasks task1 task2

# Run with specific models
python run_evaluation.py --models openai gemini
```

---

## License

This dataset is released under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

- The causal triplets and benchmark tasks are derived from peer-reviewed academic papers.
- The source papers are the intellectual property of their respective authors and publishers.
- This dataset is intended for **research purposes only**.

---

## Citation

If you use EconCausal in your research, please cite:

```bibtex
@article{lee2025econcausal,
  title={EconCausal: A Context-Aware Causal Reasoning Benchmark for Large Language Models in Social Science},
  author={Lee, Donggyu and Yun, Hyeok and Cha, Meeyoung and Park, Sungwon and Park, Sangyoon and Kim, Jihee},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## Contact

For questions or feedback, send mail to donggyu.lee@kaist.ac.kr
