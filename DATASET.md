# EconCausal Dataset

## 1. Overview

EconCausal is a context-aware causal reasoning dataset designed to evaluate whether large language models (LLMs) can correctly infer causal directions under varying socio-economic contexts in economics and social science.

The dataset represents causal knowledge using a structured four-element formulation:
- T (Treatment): the intervention or explanatory variable
- O (Outcome): the affected economic or social outcome
- D (Direction): the reported causal sign
- C (Context): institutional, temporal, and geographic conditions

All causal claims are grounded in peer-reviewed empirical research and reflect the headline findings reported by the original authors.

⸻

## 2. Intended Use

This dataset is intended for:
- Benchmarking context-dependent causal reasoning capabilities of LLMs
- Evaluating robustness to context shifts and counterfactual settings
- Research on AI systems for economics, finance, and social science

This dataset is not intended for:
- Automated policy decision-making
- Estimation of causal effect sizes
- Use in real-world regulatory or legal settings without expert verification

⸻

## 3. Data Sources

EconCausal is constructed from 4,176 peer-reviewed empirical studies, primarily drawn from:
- NBER Working Papers (published versions)
- Top-tier economics journals (AER, QJE, JPE, Review of Economic Studies, Econometrica)
- Top-tier finance journals (Journal of Finance, Journal of Financial Economics, Review of Financial Studies)

Only studies with explicit econometric identification strategies are included, such as:
- Difference-in-Differences
- Instrumental Variables
- Regression Discontinuity Designs
- Randomized Controlled Trials

⸻

## 4. Dataset Construction Pipeline

The dataset is produced through a four-stage pipeline.
Intermediate outputs are included to ensure transparency.

### Step 1. Raw Causal Extraction

File: data/step1_raw_extractions.csv
- Candidate treatment–outcome pairs extracted using large language models
- Output is intentionally over-inclusive and noisy

### Step 2. Paper-Level Metadata Annotation

File: data/step2_paper_metadata.csv
- Extraction of paper-level attributes including identification strategy
- Annotation of time period, geographic scope, and institutional setting

### Step 3. Context-Matched Triplet Construction

File: data/step3_matched_triplets.csv
- Alignment of causal claims with consistent contextual descriptions
- Removal of ambiguous or weakly grounded claims

### Step 4. Multi-Critic Quality Control

File: data/step4_final_benchmark.csv
- Filtering based on clarity of causal direction
- Consistency with identification strategy
- Context specificity and absence of speculative language

This file constitutes the final benchmark dataset.

⸻

## 5. Annotation Scheme

Each entry in the final benchmark contains:
- Treatment: textual description of the intervention
- Outcome: textual description of the outcome
- Direction:
	•	Positive (+)
	•	Negative (−)
	•	None (no statistically significant effect)
	•	Mixed (heterogeneous or context-dependent effects)
- Context:
	•	Country or region
	•	Time period
	•	Institutional or policy environment

The direction label reflects the authors’ preferred specification rather than re-estimated effects.

⸻

## 6. Dataset Statistics
- Total causal tuples: 10,439
- Source papers: 4,176
- Coverage: more than 20 JEL categories
- Average tuples per paper: approximately 2.5

Detailed statistics are reported in the accompanying paper.

⸻

## 7. Supported Benchmark Tasks

The dataset supports three benchmark tasks:
	1.	Causal Sign Identification
	2.	Context-Dependent Sign Prediction
	3.	Misinformation-Robust Reasoning

Each task is designed to evaluate reasoning beyond surface-level pattern matching.

⸻

## 8. Known Limitations
- Numerical effect sizes are not provided
- Context descriptions are textual and vary in granularity
- The dataset reflects published empirical findings and may inherit publication bias
- Mixed labels may aggregate heterogeneous mechanisms

Users should interpret benchmark results with these limitations in mind.

⸻

## 9. Legal and Usage Notes
- All data are derived from publicly available academic research
- Original paper PDFs are not redistributed due to copyright restrictions
- The dataset is intended strictly for research and benchmarking purposes

⸻

## 10. Citation

If you use this dataset, please cite the accompanying EconCausal paper.
Citation metadata is provided in the CITATION.cff file.

⸻

## 11. Contact

For questions or issues related to the dataset, please open a GitHub issue or contact the authors listed in the paper.
