"""JSON schemas for structured outputs from each pipeline step."""

# Step 1: Causal relations extraction schema
STEP1_SCHEMA = {
    "name": "causal_relations",
    "schema": {
        "type": "object",
        "properties": {
            "causal_relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "treatment": {
                            "type": "string",
                            "description": "A concise noun phrase representing the independent variable or intervention (under 10 words)"
                        },
                        "outcome": {
                            "type": "string",
                            "description": "A concise noun phrase representing the dependent variable or affected endpoint (under 10 words)"
                        },
                        "sign": {
                            "type": "string",
                            "enum": ["+", "-", "None", "mixed"],
                            "description": "Direction of causal effect: + (positive), - (negative), None (no significant effect), mixed (heterogeneous)"
                        },
                        "supporting_evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Up to three verbatim paragraphs from the paper supporting this causal claim"
                        }
                    },
                    "required": ["treatment", "outcome", "sign", "supporting_evidence"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["causal_relations"],
        "additionalProperties": False
    }
}

# Step 2: Paper metadata and context schema
STEP2_SCHEMA = {
    "name": "paper_metadata",
    "schema": {
        "type": "object",
        "properties": {
            "paper_metadata": {
                "type": "object",
                "properties": {
                    "paper_type": {
                        "type": "string",
                        "enum": ["empirical", "theoretical"],
                        "description": "Classification of paper type"
                    }
                },
                "required": ["paper_type"],
                "additionalProperties": False
            },
            "context": {
                "type": ["string", "null"],
                "description": "Paper-wide context paragraph (max 100 words) summarizing when, where, who, and background"
            },
            "identification_methods": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "RDD",
                        "IV",
                        "RCT",
                        "DiD",
                        "event studies",
                        "synthetic control",
                        "PSM",
                        "other panel regressions",
                        "other time-series regressions",
                        "other cross-sectional regressions",
                        "others"
                    ]
                },
                "description": "List of identification methods used in the paper"
            }
        },
        "required": ["paper_metadata", "context", "identification_methods"],
        "additionalProperties": False
    }
}

# Step 3: Context selection schema
STEP3_SCHEMA = {
    "name": "context_selection",
    "schema": {
        "type": "object",
        "properties": {
            "selection": {
                "type": "object",
                "properties": {
                    "context_selected": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Selected or minimally rewritten context items applicable to this triplet"
                    },
                    "id_method_selected": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Selected identification methods applicable to this triplet"
                    }
                },
                "required": ["context_selected", "id_method_selected"],
                "additionalProperties": False
            }
        },
        "required": ["selection"],
        "additionalProperties": False
    }
}

# Step 4: Critic evaluation schema
STEP4_SCHEMA = {
    "name": "critic_evaluation",
    "schema": {
        "type": "object",
        "properties": {
            "triplet": {
                "type": "object",
                "properties": {
                    "treatment": {"type": "string"},
                    "outcome": {"type": "string"},
                    "sign": {
                        "type": "string",
                        "enum": ["+", "-", "None", "mixed"]
                    }
                },
                "required": ["treatment", "outcome", "sign"],
                "additionalProperties": False
            },
            "scores": {
                "type": "object",
                "properties": {
                    "variable_extraction": {
                        "type": "integer",
                        "enum": [0, 1, 2, 3],
                        "description": "Score for treatment and outcome variable extraction"
                    },
                    "direction": {
                        "type": "integer",
                        "enum": [0, 1, 2, 3],
                        "description": "Score for correct direction (T -> O)"
                    },
                    "sign": {
                        "type": "integer",
                        "enum": [0, 1, 2, 3],
                        "description": "Score for sign accuracy"
                    },
                    "causality": {
                        "type": "integer",
                        "enum": [0, 1, 2, 3],
                        "description": "Score for whether this is a true causal claim"
                    },
                    "main_claim": {
                        "type": "integer",
                        "enum": [0, 1, 2, 3],
                        "description": "Score for whether this is a core/headline causal claim of the paper"
                    }
                },
                "required": ["variable_extraction", "direction", "sign", "causality", "main_claim"],
                "additionalProperties": False
            },
            "reasons": {
                "type": "object",
                "properties": {
                    "variable_extraction": {
                        "type": "string",
                        "description": "One-sentence justification for variable_extraction score"
                    },
                    "direction": {
                        "type": "string",
                        "description": "One-sentence justification for direction score"
                    },
                    "sign": {
                        "type": "string",
                        "description": "One-sentence justification for sign score"
                    },
                    "causality": {
                        "type": "string",
                        "description": "One-sentence justification for causality score"
                    },
                    "main_claim": {
                        "type": "string",
                        "description": "One-sentence justification for main_claim score"
                    }
                },
                "required": ["variable_extraction", "direction", "sign", "causality", "main_claim"],
                "additionalProperties": False
            }
        },
        "required": ["triplet", "scores", "reasons"],
        "additionalProperties": False
    }
}

# Step 4 with context evaluation: Critic evaluation schema (includes context_appropriateness)
STEP4_context_SCHEMA = {
    "name": "critic_evaluation_with_context",
    "schema": {
        "type": "object",
        "properties": {
            "triplet": {
                "type": "object",
                "properties": {
                    "treatment": {"type": "string"},
                    "outcome": {"type": "string"},
                    "sign": {
                        "type": "string",
                        "enum": ["+", "-", "None", "mixed"]
                    }
                },
                "required": ["treatment", "outcome", "sign"],
                "additionalProperties": False
            },
            "scores": {
                "type": "object",
                "properties": {
                    "variable_extraction": {
                        "type": "integer",
                        "enum": [0, 1, 2, 3],
                        "description": "Score for treatment and outcome variable extraction"
                    },
                    "direction": {
                        "type": "integer",
                        "enum": [0, 1, 2, 3],
                        "description": "Score for correct direction (T -> O)"
                    },
                    "sign": {
                        "type": "integer",
                        "enum": [0, 1, 2, 3],
                        "description": "Score for sign accuracy"
                    },
                    "causality": {
                        "type": "integer",
                        "enum": [0, 1, 2, 3],
                        "description": "Score for whether this is a true causal claim"
                    },
                    "main_claim": {
                        "type": "integer",
                        "enum": [0, 1, 2, 3],
                        "description": "Score for whether this is a core/headline causal claim of the paper"
                    },
                    "context_appropriateness": {
                        "type": "integer",
                        "enum": [0, 1, 2, 3],
                        "description": "Score for whether the context is appropriate and complete for this triplet"
                    }
                },
                "required": ["variable_extraction", "direction", "sign", "causality", "main_claim", "context_appropriateness"],
                "additionalProperties": False
            },
            "reasons": {
                "type": "object",
                "properties": {
                    "variable_extraction": {
                        "type": "string",
                        "description": "One-sentence justification for variable_extraction score"
                    },
                    "direction": {
                        "type": "string",
                        "description": "One-sentence justification for direction score"
                    },
                    "sign": {
                        "type": "string",
                        "description": "One-sentence justification for sign score"
                    },
                    "causality": {
                        "type": "string",
                        "description": "One-sentence justification for causality score"
                    },
                    "main_claim": {
                        "type": "string",
                        "description": "One-sentence justification for main_claim score"
                    },
                    "context_appropriateness": {
                        "type": "string",
                        "description": "One-sentence justification for context_appropriateness score"
                    }
                },
                "required": ["variable_extraction", "direction", "sign", "causality", "main_claim", "context_appropriateness"],
                "additionalProperties": False
            }
        },
        "required": ["triplet", "scores", "reasons"],
        "additionalProperties": False
    }
}


# ==================== Evaluation Task Schemas ====================

# Task 1: Causality Verification Schema
EVAL_TASK1_SCHEMA = {
    "name": "causality_verification",
    "schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "enum": ["Yes", "No"],
                "description": "Whether the causal claim is true under the given context"
            },
            "reasoning": {
                "type": "string",
                "description": "Step-by-step reasoning for the answer (max 200 words)"
            }
        },
        "required": ["answer", "reasoning"],
        "additionalProperties": False
    }
}

# Task 2: Sign Prediction Schema
EVAL_TASK2_SCHEMA = {
    "name": "sign_prediction",
    "schema": {
        "type": "object",
        "properties": {
            "predicted_sign": {
                "type": "string",
                "enum": ["+", "-", "None", "mixed"],
                "description": "Predicted direction of causal effect"
            },
            "reasoning": {
                "type": "string",
                "description": "Economic reasoning for the predicted sign"
            }
        },
        "required": ["predicted_sign", "reasoning"],
        "additionalProperties": False
    }
}

# Task 2: Sign Prediction Schema (with unknown option)
EVAL_TASK2_UNKNOWN_SCHEMA = {
    "name": "sign_prediction",
    "schema": {
        "type": "object",
        "properties": {
            "predicted_sign": {
                "type": "string",
                "enum": ["+", "-", "None", "mixed", "unknown"],
                "description": "Predicted direction of causal effect"
            },
            "reasoning": {
                "type": "string",
                "description": "Economic reasoning for the predicted sign"
            }
        },
        "required": ["predicted_sign", "reasoning"],
        "additionalProperties": False
    }
}

# Task 3: Context-Aware Reasoning (Treatment/Outcome Fixed) Schema
EVAL_TASK3_SCHEMA = {
    "name": "context_aware_to_fixed",
    "schema": {
        "type": "object",
        "properties": {
            "predicted_sign": {
                "type": "string",
                "enum": ["+", "-", "None", "mixed"],
                "description": "Predicted sign for the new context"
            },
            "context_analysis": {
                "type": "string",
                "description": "How the new context differs from the examples"
            },
            "reasoning": {
                "type": "string",
                "description": "Why the predicted sign follows from the context differences"
            }
        },
        "required": ["predicted_sign", "context_analysis", "reasoning"],
        "additionalProperties": False
    }
}

# Task 4: Context-Aware Reasoning (Context Fixed) Schema
EVAL_TASK4_SCHEMA = {
    "name": "context_aware_context_fixed",
    "schema": {
        "type": "object",
        "properties": {
            "predicted_sign": {
                "type": "string",
                "enum": ["+", "-", "None", "mixed"],
                "description": "Predicted sign for the new treatment-outcome pair"
            },
            "pattern_analysis": {
                "type": "string",
                "description": "Identified causal patterns within this context"
            },
            "reasoning": {
                "type": "string",
                "description": "Why the new T/O pair should have this sign given the context"
            }
        },
        "required": ["predicted_sign", "pattern_analysis", "reasoning"],
        "additionalProperties": False
    }
}
