"""Prompts for each step of the causal relation extraction pipeline."""

# Step 1: Extract causal relations from paper
STEP1_PROMPT = """# Role

You are an expert Research Assistant in Economics and Finance, specialized in causal inference and systematic literature reviews.



# Goal

From the provided economics paper text, extract all causal effect claims explicitly asserted by the authors, and return them as a JSON array of objects.



# Definitions & Scope

- **Causal Claim**: A statement in which the authors argue that a change in X (Treatment) causes a change in Y (Outcome).

- **Strict Exclusion**: Exclude simple correlations, descriptive statistics, predictive relationships, or purely theoretical conjectures.

- **Identification Focus**: Prioritize claims supported by empirical identification strategies (e.g., Difference-in-Differences, Instrumental Variables, Regression Discontinuity, Fixed Effects, or Randomized Controlled Trials).

- **No Context at This Stage**: Extract only the treatment and outcome variables as abstract concepts. Do **not** include who/when/where, population, or institutional context; these will be extracted in a later step.

- **Expand Shorthand References**: If treatments or outcomes are referred to using pronouns, acronyms, abbreviations, or shorthand/composite labels (e.g., "it", "the program", "JP", "joint program") after being defined earlier in the paper, always recover and spell out the fully specified variable as originally defined.



# Extraction Rules

1. **treatment**: A concise noun phrase representing the independent variable or intervention (e.g., "Minimum wage increase").

2. **outcome**: A concise noun phrase representing the dependent variable or affected endpoint (e.g., "Teenage employment rate").

3. **sign**: Categorize the direction of the causal effect as follows:

   - '+': The authors' preferred / main estimate shows that the Treatment increases the Outcome and is statistically significant.

   - '-': The authors' preferred / main estimate shows that the Treatment decreases the Outcome and is statistically significant.

   - 'None': Under the authors' preferred / main specification, no statistically significant effect is found.

   - 'mixed': The effect is heterogeneous, or the direction truly varies across subgroups, or the authors present multiple equally central results with opposite signs.



   **Priority / tie-break rule**:

   - Identify the authors' preferred / baseline / main result (explicitly labeled as such, or summarized as the headline finding in the Abstract or Conclusion).

   - Assign the sign based **only** on this main result: use '+' or '-' only if it is statistically significant; otherwise use 'None'.

   - Use 'mixed' only when the paper's own core interpretation is that the effect differs in direction across subgroups (e.g., positive for one group and negative for another), or when equally emphasized headline results have conflicting signs.

   - If the direction is stable but statistical significance varies across robustness checks or specifications, do **not** label as 'mixed'; follow the preferred / main specification's direction and significance.



4. **supporting_evidence**: Extract up to **three** paragraphs verbatim that report the main results, coefficient estimates, or causal conclusions for this specific treatment–outcome pair.



# Constraints

- **Verbatim Only**: Supporting paragraphs must be copied exactly from the source text, without paraphrasing.

- **Short Phrases**: Keep the treatment and outcome fields under 10 words each.

- **No Hallucination**: If no explicit causal claims are identified, return an empty array [].

- **Output Format**: Return **only** a valid JSON array. Do not include markdown code blocks (unless explicitly requested), preambles, or conversational commentary.



# JSON Schema Example



[

  {

    "treatment": "Minimum wage increase",

    "outcome": "Full-time equivalent employment",

    "sign": "+",

    "supporting_evidence": [

"On April 1, 1992, New Jersey's minimum wage rose from $4.25 to $5.05 per hour. To evaluate the impact of the law we surveyed 410 fast-food restaurants in New Jersey and eastern Pennsylvania before and after the rise. Comparisons of employment growth at stores in New Jersey and Pennsylvania (where the minimum wage was constant) provide simple estimates of the effect of the higher minimum wage. We also compare employment changes at stores in New Jersey that were initially paying high wages (above $5) to the changes at lower-wage stores. We find no indication that the rise in the minimum wage reduced employment.",

"As noted in Table 2, New Jersey stores were initially smaller than their Pennsylvania counterparts but grew relative to Pennsylvania stores after the rise in the minimum wage. The relative gain (the \"difference in differences\" of the changes in employment) is 2.76 FTE employees (or 13 percent), with a t statistic of 2.03. Inspection of the averages in rows 4 and 5 shows that the relative change between New Jersey and Pennsylvania stores is virtually identical when the analysis is restricted to the unbalanced subsample, and it is only slightly smaller when wave-2 employment at the temporarily closed stores is treated as zero."

    ]

  },

  {

    "treatment": "Minimum wage increase",

    "outcome": "Meal prices",

    "sign": "+",

    "supporting_evidence": [

      "The estimated New Jersey dummy in column (i) shows that after-tax meal prices rose 3.2-percent faster in New Jersey than in Pennsylvania between February and November 1992.",

      "Finally, we find that prices of fast-food meals increased in New Jersey relative to Pennsylvania, suggesting that much of the burden of the minimum-wage rise was passed on to consumers."

    ]

  },

  {

    "treatment": "Minimum wage increase",

    "outcome": "Fringe benefits",

    "sign": "None",

    "supporting_evidence": [

      "The results provide no evidence that employers offset the minimum-wage increase by reducing free or reduced-price meals.",

      "However, the relative shift is not statistically significant.",

      "In summary, we can find no indication that New Jersey employers changed either their fringe benefits or their wage profiles to offset the rise in the minimum wage."

    ]

  }

]



# Text to Process

Please analyze the provided PDF document."""


# Step 2: Extract paper metadata and context
STEP2_PROMPT = """# Role

You are an expert Research Assistant in Economics.

# Goal

Your task is to extract paper-level metadata, research context, and identification strategies into a structured JSON format.



Task Instructions:



1. Metadata:

   - Classify the paper type as either "empirical" or "theoretical".

     * empirical: uses data/estimation to make causal claims

     * theoretical: develops models or theory without empirical estimation



2. Global Context:



- Produce ONE cohesive natural-language paragraph (maximum 100 words) summarizing the **paper-wide context underlying the main causal claims**.

- The paragraph should reflect the **union of contexts** used across the paper's major causal analyses.



The paragraph should integrate, when available:

- **When**: time period, years, cohorts, event timing, or historical window studied

- **Where**: geography, country/region, market, industry, platform, or institutional setting

- **Who / Unit**: unit of observation (individuals, households, firms, regions, products, transactions, etc.)

- **Background**: key policy environment, institutional details, economic setting, or core assumptions

- Additionally, include any other important paper-wide contextual elements if present (e.g., eligibility rules, regulatory regimes, macro shocks, industry structure), while keeping the paragraph under 100 words.



3. Identification Methods

- Identify **ALL empirical identification methods that are applied to support the paper's causal claims**.

- Include methods used in main analyses or robustness checks **only if they contribute to identifying a causal effect**.

- Exclude purely descriptive, predictive, or exploratory analyses that are not tied to causal claims.

- Deduplicate methods.



You MUST map each method strictly to one of the following allowed values:

     [DiD, IV, RCT, RDD, event studies, synthetic control, PSM, other panel regressions, other time-series regressions, other cross-sectional regressions, others]

   Mapping Rules:

   - DiD: Difference-in-differences, staggered adoption, TWFE DiD

   - IV: Instrumental variables, 2SLS, shift-share IV

   - RCT: Randomized controlled trials, field or lab experiments

   - RDD: Regression discontinuity designs

   - event studies: Dynamic DiD, leads-and-lags, event-time coefficients

   - synthetic control: Synthetic control or generalized synthetic control methods

   - PSM: Propensity score matching, propensity score

   - other panel regressions: panel regressions that are not listed above

   - other time-series regressions: time-series analysis methods

   - other cross-sectional regressions: cross-sectional regression methods

   - others: Matching, structural estimation, simulations, or methods not listed above

   - If the paper is purely theoretical, return an empty list [].



Output Format (STRICT JSON ONLY):

{

  "paper_metadata": {

    "empirical"

  },

  "context": "This study analyzes 410 fast-food restaurants (Burger King, KFC, Wendy's, Roy Rogers) in New Jersey and eastern Pennsylvania during 1992. The institutional setting involves New Jersey raising its state minimum wage from $4.25 to $5.05 per hour on April 1, 1992, while Pennsylvania's minimum wage remained constant. This policy change occurred during an economic recession. The authors surveyed establishments before (February-March) and after (November-December) the increase to evaluate the labor market and pricing responses within the low-wage fast-food industry.",

  "identification_methods": [

    "DiD"

  ]

}



Constraints:

- Output ONLY valid JSON (no markdown, no commentary).

- If information is missing or cannot be determined, use null (or [] for lists).

- If paper_type is "theoretical", identification_strategies MUST be [].



Input:

Please analyze the provided PDF document."""


# Step 3: Select context for each triplet
STEP3_PROMPT = """# Role

You are an expert Research Assistant in Economics.

# Goal

Your task is to perform **LOCAL REVIEW ONLY** for a single extracted causal triplet.

You are given:
- An extracted causal triplet (treatment, outcome, sign)
- Evidence paragraph(s) supporting this triplet (verbatim)
- A paper-wide **global context paragraph** (a single, coherent natural-language summary, ~100 words)
- A paper-wide list of **identification methods** used in the paper
- The original economics paper from which the causal triplet and global context were extracted

IMPORTANT
- These global items may be **broader or more general** than what applies to this specific triplet.
- Your job is **NOT** to judge whether the causal triplet itself is correct.

────────────────────────────────────────
TASK: LOCAL REVIEW

A) Context review

- Using the **original paper** (relevant sections), infer the **triplet-specific context** and compare it to `global_context`.

- Default: return `[]` (keep `global_context`).
- Rewrite `global_context` into a triplet-specific paragraph ONLY if the paper clearly shows that `global_context` is NOT valid for this triplet as-is, because:
  - (a) the triplet’s context is **more specific or different** (time / place / unit / background), OR
  - (b) some elements in `global_context` are **not applicable / not supported** for this triplet.

- If rewriting, apply the **minimum-edit rule**:
  - **Keep** the parts of `global_context` that still apply.
  - **Remove or adjust** only the parts that do not apply.
  - **Add or narrow** details only when **explicitly supported** for this triplet.

- Output format (if rewriting):
  - ONE cohesive natural-language paragraph (≤ 100 words).
  - Summarize the context underlying THIS causal triplet, consistent with the paper’s overall framing.
  - Include only what the paper explicitly supports:
    - **When**: time period, years, cohorts, event timing
    - **Where**: geography, country/region, market, industry, institutional setting
    - **Who / Unit**: unit of observation (individuals, households, firms, regions, etc.)
    - **Background**: policy environment, institutional details, economic setting
  - Add other explicitly stated context if needed (e.g., eligibility rules, regulatory regimes, macro shocks).
  - Do NOT exceed 100 words.



B) Identification-method review

- Compare the identification method implied for THIS triplet (from the evidence paragraph(s) and relevant paper sections) against `global_identification_method`.

- Default: return `[]` (keep `global_identification_method`).
- Output a non-empty list ONLY if the evidence explicitly shows that the triplet uses an identification method that is:
  - different from `global_identification_method`, OR
  - a stricter/more specific version of it, OR
  - missing from it (i.e., clearly used for this triplet but not listed globally).

- If updating, list ONLY the corrected/missing method(s), each mapped to EXACTLY ONE allowed value:
  [DiD, IV, RCT, RDD, event studies, synthetic control, PSM, other panel regressions, other time-series regressions, other cross-sectional regressions, others]

- Mapping Rules:
  - DiD: difference-in-differences, staggered adoption, TWFE DiD
  - IV: instrumental variables, 2SLS, shift-share IV
  - RCT: randomized controlled trials, field/lab experiments
  - RDD: regression discontinuity designs
  - event studies: dynamic DiD, leads-and-lags, event-time coefficients
  - synthetic control: synthetic control, generalized synthetic control
  - PSM: propensity score matching
  - other panel regressions: panel regressions not covered above
  - other time-series regressions: time-series regressions not covered above
  - other cross-sectional regressions: cross-sectional regressions not covered above
  - others: matching/structural/simulation or any method not listed above


────────────────────────────────────────
STRICT RULES

- Use ONLY the provided data.
- Do NOT guess, infer, or extrapolate beyond explicit textual support.
- Do NOT restate or paraphrase global items unless a clear mismatch is present.
- If there is no clear evidence-based difference, return empty lists.
- Output MUST be valid JSON and NOTHING ELSE.

Required Output

{{
  "selection": {{
    "context_selected": [string, ...],      // rewritten ONLY if evidence clearly differs; otherwise []
    "id_method_selected": [string, ...]     // rewritten/added ONLY if evidence clearly differs; otherwise []
  }}
}}

INPUTS

Extracted causal triplet:
- treatment (T): {treatment}
- outcome (O): {outcome}
- sign: {sign}

Paper-wide summaries:
- global_context: {global_context}
- global_identification_method: {global_identification_method}

Evidence paragraph(s) (verbatim):
{evidence_paragraphs}

Original paper:
(See the attached PDF document)
"""



STEP4_context_PROMPT = """
# Role

You are a **causal-inference critic** reviewing an extracted causal relation from an Economics/Finance paper.

# Goal 

Given an extracted **causal triplet** and supporting materials, evaluate whether the triplet is correctly extracted and score it on **six criteria (0–3 each)** using **only** the provided inputs.

# Inputs You Will Receive

1) Extracted triplet:
- treatment (T)
- outcome (O)
- sign  // one of ["+", "-", "None", "mixed"]

2) Evidence paragraph(s) for this triplet (verbatim)

3) Selected context for this triplet (as provided by the extraction pipeline)

4) Original paper (full text)


# Extraction Rules (how the triplet was originally extracted)

## Triplet Extraction Rules

1. **treatment**: A concise noun phrase representing the independent variable or intervention (e.g., "Minimum wage increase").

2. **outcome**: A concise noun phrase representing the dependent variable or affected endpoint (e.g., "Teenage employment rate").

3. **sign**: Categorize the direction of the causal effect as follows:

   - '+': The authors' preferred (baseline/main) estimate shows that the Treatment increases the Outcome and is statistically significant.

   - '-': The authors' preferred (baseline/main) estimate shows that the Treatment decreases the Outcome and is statistically significant.

   - 'None': Under the authors' preferred (baseline/main) specification, no statistically significant effect is found.

   - 'mixed': The effect is heterogeneous, or the direction truly varies across subgroups, or the authors present multiple equally central results with opposite signs.


## Context Extraction Rules

The selected context paragraph should integrate, **when available**, the following paper-wide elements:

- **When**: time period, years, cohorts, event timing, or historical window studied  
- **Where**: geography, country/region, market, industry, platform, or institutional setting  
- **Who / Unit**: unit of observation (e.g., individuals, households, firms, regions, products, transactions)  
- **Background**: key policy environment, institutional details, economic setting, or core assumptions  

Additionally, include any other important contextual elements if present (e.g., eligibility rules, regulatory regimes, macro shocks, industry structure), while keeping the paragraph **under 100 words** and **avoiding redundancy or causal interpretation**.


# Scoring (0–3)

Use this rubric for every dimension:

- **3** = clearly supported / correct  
- **2** = mostly supported / minor ambiguity  
- **1** = weak support / substantial ambiguity / could be mis-specified  
- **0** = not supported / contradicted / clearly wrong  

For EACH scored dimension (1–6), you MUST provide a one-sentence justification explaining why that score was assigned.


# Dimensions to Score

1) **variable_extraction**
- Are **treatment** and **outcome** extracted as **concise, concrete noun phrases** that are explicitly mentioned or defined in the paper?
- If the paper uses pronouns, acronyms, abbreviations, or shorthand references (e.g., “it,” “the program,” “JP”), are these **correctly expanded to the fully specified variables** as originally defined in the paper?

2) **direction**
- Does the triplet correctly capture the **intended causal direction** (Treatment → Outcome) asserted by the authors without reversal due to ambiguous or easily flipped wording?

3) **sign**
- Does the **sign** (+/−/None/mixed) match the authors’ **preferred/baseline/main estimate** described in the evidence (not a robustness check or secondary specification)?
- Does it follow the rule: use **+/- only if statistically significant**, **None** if not significant under the preferred specification, and **mixed** only for truly heterogeneous or opposite-direction **headline results**?

4) **causality**
- Is the extracted relationship presented as a **causal effect claim** supported by an identification strategy?
- Does it **exclude** simple correlations/associations, descriptive statistics, predictive relationships, or purely theoretical conjectures?

5) **main_claim**
- Is the triplet a **core causal claim** of the paper emphasized by the authors (e.g., a headline or central contribution in the abstract, introduction, or conclusion), rather than a secondary, peripheral, or incidental finding?

6) **context_appropriateness**
- Does the context include the key elements required to interpret the causal claim, without omitting paper-critical setting information?
- Does the context avoid encoding or implying the correctness of the triplet, including its sign, direction, or causal validity?



# Required Output

Return **one valid JSON object only** following this schema:

{{
  "triplet": {{
    "treatment": "string",
    "outcome": "string",
    "sign": "+" | "-" | "None" | "mixed"
  }},
  "scores": {{
    "variable_extraction": 0 | 1 | 2 | 3,
    "direction": 0 | 1 | 2 | 3,
    "sign": 0 | 1 | 2 | 3,
    "causality": 0 | 1 | 2 | 3,
    "main_claim": 0 | 1 | 2 | 3,
    "context_appropriateness": 0 | 1 | 2 | 3
  }},
  "reasons": {{
    "variable_extraction": "string (one-sentence justification)",
    "direction": "string (one-sentence justification)",
    "sign": "string (one-sentence justification)",
    "causality": "string (one-sentence justification)",
    "main_claim": "string (one-sentence justification)",
    "context_appropriateness": "string (one-sentence justification)"
  }}
}}

# INPUTS

Extracted triplet:
- treatment (T): {treatment}
- outcome (O): {outcome}
- sign: {sign}

Evidence paragraph(s) (verbatim):
{evidence_paragraphs}

Selected context for this triplet:
{context_selected}

Original paper:
(See the attached PDF document)
"""

# Step 4: Critic evaluation
STEP4_PROMPT = """
# Role

You are a **causal-inference critic** reviewing an extracted causal relation from an Economics/Finance paper.

# Goal 

Given an extracted **causal triplet** and supporting materials, evaluate whether the triplet is correctly extracted and score it on **five criteria (0–3 each)** using **only** the provided inputs.

# Inputs You Will Receive

1) Extracted triplet:
- treatment (T)
- outcome (O)
- sign  // one of ["+", "-", "None", "mixed"]

2) Evidence paragraph(s) for this triplet (verbatim)

3) Selected context for this triplet (as provided by the extraction pipeline)

4) Original paper (full text)


# Extraction Rules (how the triplet was originally extracted)

1. **treatment**: A concise noun phrase representing the independent variable or intervention (e.g., "Minimum wage increase").

2. **outcome**: A concise noun phrase representing the dependent variable or affected endpoint (e.g., "Teenage employment rate").

3. **sign**: Categorize the direction of the causal effect as follows:

   - '+': The authors' preferred (baseline/main) estimate shows that the Treatment increases the Outcome and is statistically significant.

   - '-': The authors' preferred (baseline/main) estimate shows that the Treatment decreases the Outcome and is statistically significant.

   - 'None': Under the authors' preferred (baseline/main) specification, no statistically significant effect is found.

   - 'mixed': The effect is heterogeneous, or the direction truly varies across subgroups, or the authors present multiple equally central results with opposite signs.


# Scoring (0–3)

Use this rubric for every dimension:

- **3** = clearly supported / correct  
- **2** = mostly supported / minor ambiguity  
- **1** = weak support / substantial ambiguity / could be mis-specified  
- **0** = not supported / contradicted / clearly wrong  

For EACH scored dimension (1–5), you MUST provide a one-sentence justification explaining why that score was assigned.


# Dimensions to Score

1) **variable_extraction**
- Are **treatment** and **outcome** extracted as **concise, concrete noun phrases** that are explicitly mentioned or defined in the paper?
- If the paper uses pronouns, acronyms, abbreviations, or shorthand references (e.g., “it,” “the program,” “JP”), are these **correctly expanded to the fully specified variables** as originally defined in the paper?

2) **direction**
- Does the triplet correctly capture the **intended causal direction** (Treatment → Outcome) asserted by the authors without reversal due to ambiguous or easily flipped wording?

3) **sign**
- Does the **sign** (+/−/None/mixed) match the authors’ **preferred/baseline/main estimate** described in the evidence (not a robustness check or secondary specification)?
- Does it follow the rule: use **+/- only if statistically significant**, **None** if not significant under the preferred specification, and **mixed** only for truly heterogeneous or opposite-direction **headline results** (not mere sensitivity to alternative specifications)?

4) **causality**
- Is the extracted relationship presented as a **causal effect claim** supported by an identification strategy?
- Does it **exclude** simple correlations/associations, descriptive statistics, predictive relationships, or purely theoretical conjectures?

5) **main_claim**
- Is the triplet a **core causal claim** of the paper emphasized by the authors (e.g., a headline or central contribution in the abstract, introduction, or conclusion), rather than a secondary, peripheral, or incidental finding?



# Required Output

Return **one valid JSON object only** following this schema:

{{
  "triplet": {{
    "treatment": "string",
    "outcome": "string",
    "sign": "+" | "-" | "None" | "mixed"
  }},
  "scores": {{
    "variable_extraction": 0 | 1 | 2 | 3,
    "direction": 0 | 1 | 2 | 3,
    "sign": 0 | 1 | 2 | 3,
    "causality": 0 | 1 | 2 | 3,
    "main_claim": 0 | 1 | 2 | 3
  }},
  "reasons": {{
    "variable_extraction": "string (one-sentence justification)",
    "direction": "string (one-sentence justification)",
    "sign": "string (one-sentence justification)",
    "causality": "string (one-sentence justification)",
    "main_claim": "string (one-sentence justification)"
  }}
}}

# INPUTS

Extracted triplet:
- treatment (T): {treatment}
- outcome (O): {outcome}
- sign: {sign}

Evidence paragraph(s) (verbatim):
{evidence_paragraphs}

Selected context for this triplet:
{context_selected}

Original paper:
(See the attached PDF document)
"""


# ==================== Evaluation Task Prompts ====================

# Task 1: Causality Verification Prompt
EVAL_TASK1_PROMPT = """# Role
You are an expert economist evaluating causal claims from empirical research.

# Task
Given a specific context and a causal claim, determine whether the claim is valid.

# Context
{context}

# Causal Claim
Treatment: {treatment}
Sign: {sign} (where + means "increases", - means "decreases", None means "no significant effect")
Outcome: {outcome}

In other words: Under the given context, does {treatment} {sign_description} {outcome}?

# Instructions
1. Consider the economic mechanisms that would connect treatment to outcome
2. Evaluate whether the stated sign is plausible given the context
3. Consider potential confounders or alternative explanations
4. Provide your judgment as Yes (claim is valid) or No (claim is invalid)

# Output Format
Respond with a JSON object containing:
- answer: "Yes" or "No"
- reasoning: Your step-by-step economic reasoning (max 200 words)
"""


# Task 2: Sign Prediction Prompt (No Context)
EVAL_TASK2_PROMPT_NO_CONTEXT = """# Role
You are an expert economist.

# Task
Given a treatment–outcome pair, predict the most likely sign of the causal effect.

# Sign Definitions
- '+': The treatment increases the outcome (positive and statistically significant effect).
- '-': The treatment decreases the outcome (negative and statistically significant effect).
- 'None': No statistically significant effect.
- 'mixed': The effect varies across subgroups or specifications.

# Treatment–Outcome Pair
Treatment: {treatment}
Outcome: {outcome}

# Output Format
Respond with a JSON object containing:
- predicted_sign: "+", "-", "None", or "mixed"
- reasoning: A concise explanation of your reasoning
"""

# Task 2: Sign Prediction Prompt (with unknown option)
EVAL_TASK2_UNKNOWN_PROMPT = """# Role
You are an expert economist.

# Task
Given a context and a treatment–outcome pair, predict the most likely sign of the causal effect.

# Sign Definitions
- '+': The treatment increases the outcome (positive and statistically significant effect).
- '-': The treatment decreases the outcome (negative and statistically significant effect).
- 'None': No statistically significant effect.
- 'mixed': The effect varies across subgroups or specifications.
- 'unknown': You cannot determine the direction of the effect given the available information.

# Context
{context}

# Treatment–Outcome Pair
Treatment: {treatment}
Outcome: {outcome}

# Output Format
Respond with a JSON object containing:
- predicted_sign: "+", "-", "None", "mixed", or "unknown"
- reasoning: A concise explanation of your reasoning
"""

# Task 2: Sign Prediction Prompt (No Context, with unknown option)
EVAL_TASK2_UNKNOWN_PROMPT_NO_CONTEXT = """# Role
You are an expert economist.

# Task
Given a treatment–outcome pair, predict the most likely sign of the causal effect.

# Sign Definitions
- '+': The treatment increases the outcome (positive and statistically significant effect).
- '-': The treatment decreases the outcome (negative and statistically significant effect).
- 'None': No statistically significant effect.
- 'mixed': The effect varies across subgroups or specifications.
- 'unknown': You cannot determine the direction of the effect given the available information.

# Treatment–Outcome Pair
Treatment: {treatment}
Outcome: {outcome}

# Output Format
Respond with a JSON object containing:
- predicted_sign: "+", "-", "None", "mixed", or "unknown"
- reasoning: A concise explanation of your reasoning
"""
# Task 2: Sign Prediction Prompt
EVAL_TASK2_PROMPT = """# Role
You are an expert economist.

# Task
Given a context and a treatment–outcome pair, predict the most likely sign of the causal effect.

# Sign Definitions
- '+': The treatment increases the outcome (positive and statistically significant effect).
- '-': The treatment decreases the outcome (negative and statistically significant effect).
- 'None': No statistically significant effect.
- 'mixed': The effect varies across subgroups or specifications.

# Context
{context}

# Treatment–Outcome Pair
Treatment: {treatment}
Outcome: {outcome}

# Output Format
Respond with a JSON object containing:
- predicted_sign: "+", "-", "None", or "mixed"
- reasoning: A concise explanation of your reasoning
"""


# Task 3: Context-Aware Reasoning (Treatment/Outcome Fixed) Prompt
EVAL_TASK3_PROMPT = """# Role
You are an expert economist.

# Task
You are given examples in which the same or comparable treatment–outcome pair is observed across multiple contexts, potentially with different causal signs.
Predict the most likely sign for the treatment–outcome pair in the target context.

# Sign Definitions
- '+': The treatment increases the outcome (positive and statistically significant effect).
- '-': The treatment decreases the outcome (negative and statistically significant effect).
- 'None': No statistically significant effect.
- 'mixed': The effect varies across subgroups or specifications.

# Reference Examples
{examples}

# Target Context
{context_new}

# Target Treatment–Outcome Pair
Treatment: {treatment}
Outcome: {outcome}

# Output Format
Respond with a JSON object containing:
- predicted_sign: "+", "-", "None", or "mixed"
- reasoning: A concise explanation of your reasoning
"""


# Task 3: Context-Aware Reasoning (Treatment/Outcome Fixed) Prompt
EVAL_TASK4_PROMPT = """# Role
You are an expert economist.

# Task
You are given examples in which the same or comparable treatment–outcome pair is observed across multiple contexts, potentially with different causal signs.
Predict the most likely sign for the treatment–outcome pair in the target context.

# Sign Definitions
- '+': The treatment increases the outcome (positive and statistically significant effect).
- '-': The treatment decreases the outcome (negative and statistically significant effect).
- 'None': No statistically significant effect.
- 'mixed': The effect varies across subgroups or specifications.

# Reference Examples
{examples}

# Target Context
{context_new}

# Target Treatment–Outcome Pair
Treatment: {treatment}
Outcome: {outcome}

# Output Format
Respond with a JSON object containing:
- predicted_sign: "+", "-", "None", or "mixed"
- reasoning: A concise explanation of your reasoning
"""
