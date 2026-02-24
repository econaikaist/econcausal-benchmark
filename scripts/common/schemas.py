"""JSON schemas for structured outputs from evaluation tasks."""

# ==================== Evaluation Task Schemas ====================

# Task 1: Sign Prediction Schema
EVAL_TASK1_SCHEMA = {
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

# Task 1: Sign Prediction Schema (with unknown option)
EVAL_TASK1_UNKNOWN_SCHEMA = {
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

# Task 2: Context-Dependent Sign Prediction Schema
EVAL_TASK2_SCHEMA = {
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

# Task 3: Misinformation-Robust Sign Prediction Schema
EVAL_TASK3_SCHEMA = {
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
