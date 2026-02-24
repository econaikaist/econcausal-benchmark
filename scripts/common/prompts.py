"""Prompts for evaluation tasks."""


# ==================== Evaluation Task Prompts ====================

# Task 1: Sign Prediction Prompt (No Context)
EVAL_TASK1_PROMPT_NO_CONTEXT = """# Role
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

# Task 1: Sign Prediction Prompt (with unknown option)
EVAL_TASK1_UNKNOWN_PROMPT = """# Role
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

# Task 1: Sign Prediction Prompt (No Context, with unknown option)
EVAL_TASK1_UNKNOWN_PROMPT_NO_CONTEXT = """# Role
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

# Task 1: Sign Prediction Prompt
EVAL_TASK1_PROMPT = """# Role
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


# Task 2: Context-Dependent Sign Prediction Prompt
EVAL_TASK2_PROMPT = """# Role
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


# Task 3: Misinformation-Robust Sign Prediction Prompt
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
