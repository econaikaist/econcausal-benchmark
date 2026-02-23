"""
Step 5: Generate final results in Excel format.

This step combines results from steps 1-4 with NBER metadata (abs.csv, jel.csv,
published.csv, ref.csv) and generates a structured Excel file similar to
'preliminary result.xlsx'.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import re

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from common.utils import (
    load_json,
    setup_logging,
    clean_sign,
)


# Default paths
DEFAULT_OUTPUT_DIR = "/home/donggyu/econ_causality/new_data/real_data"
DEFAULT_NBER_METADATA_DIR = "/home/donggyu/econ_causality/new_data"
DEFAULT_PDF_DIR = "/home/donggyu/econ_causality/new_data/nber_paper_wo_aer_pp_30pages"

# Comprehensive pattern for ALL illegal XML 1.0 characters
# XML 1.0 only allows: #x9 (tab), #xA (newline), #xD (carriage return), #x20-#xD7FF, #xE000-#xFFFD
# This pattern removes: C0 controls (except tab/LF/CR), DEL (0x7f), C1 controls (0x80-0x9f),
# and Unicode noncharacters (U+FFFE, U+FFFF)
ILLEGAL_XML_CHARS_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffe\uffff]')

# Characters that trigger Excel formula interpretation
FORMULA_TRIGGER_CHARS = ('=', '+', '-', '@', '\t', '\r', '\n')


def escape_formula_string(value: str) -> str:
    """
    Escape strings that Excel might interpret as formulas.

    Excel interprets cells starting with =, +, -, @, tab, CR, or newline
    as formulas. We prefix with a single quote to prevent this.
    """
    if value and isinstance(value, str) and value.startswith(FORMULA_TRIGGER_CHARS):
        return "'" + value
    return value


def clean_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Remove illegal XML characters and escape formula-like strings in a DataFrame."""
    df = df.copy()
    # Columns where +/- should NOT be escaped (they are valid values, not formulas)
    no_escape_cols = {'sign'}

    for col in df.columns:
        # Check for string-like columns (dtype.kind == 'O' catches both object and string dtypes)
        if df[col].dtype.kind == 'O':
            if col in no_escape_cols:
                # Only remove illegal XML characters, don't escape formula characters
                df[col] = df[col].apply(
                    lambda x: ILLEGAL_XML_CHARS_RE.sub('', str(x)) if pd.notna(x) else x
                )
            else:
                df[col] = df[col].apply(
                    lambda x: escape_formula_string(ILLEGAL_XML_CHARS_RE.sub('', str(x))) if pd.notna(x) else x
                )
    return df


def build_paper_venue_mapping(pdf_dir: str) -> dict:
    """
    Build paper_id -> published_venue mapping from PDF folder structure.

    Args:
        pdf_dir: Directory containing journal subfolders with PDFs

    Returns:
        Dictionary mapping paper_id to journal folder name (e.g., "american_economic_review")
    """
    pdf_path = Path(pdf_dir)
    mapping = {}

    if not pdf_path.exists():
        return mapping

    for journal_folder in pdf_path.iterdir():
        if journal_folder.is_dir():
            venue_name = journal_folder.name  # e.g., "american_economic_review"
            for pdf_file in journal_folder.glob("*.pdf"):
                # Extract paper_id from filename (e.g., "0009.pdf" -> "0009")
                paper_id = pdf_file.stem
                # Remove leading zeros for consistency (e.g., "0009" -> "9")
                numeric_id = str(int(paper_id)) if paper_id.isdigit() else paper_id
                mapping[numeric_id] = venue_name
                # Also store with original format for lookup flexibility
                mapping[paper_id] = venue_name

    return mapping


def load_nber_metadata(metadata_dir: str) -> dict:
    """
    Load NBER metadata from CSV files.

    Args:
        metadata_dir: Directory containing abs.csv, jel.csv, published.csv, ref.csv

    Returns:
        Dictionary with parsed metadata:
        {
            "abstracts": {paper_id: abstract},
            "jel_codes": {paper_id: [jel_codes]},
            "published": {paper_id: {title, author, publication_year, published_venue}},
            "references": {paper_id: [{author, title, issue_date, doi}]}
        }
    """
    metadata_path = Path(metadata_dir)

    result = {
        "abstracts": {},
        "jel_codes": {},
        "published": {},
        "references": {},
    }

    # Load abs.csv (format: "w####\tAbstract text...")
    # Note: The CSV has variable number of fields due to commas in abstracts.
    # Read line by line using csv module to handle variable column counts.
    import csv
    abs_path = metadata_path / "abs.csv"
    if abs_path.exists():
        with open(abs_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header row
            for row in reader:
                # Join all columns with comma to reconstruct original text
                full_text = ",".join(row)
                if "\t" in full_text:
                    paper_id, abstract = full_text.split("\t", 1)
                    # Extract numeric part from paper_id (w18347 -> 18347)
                    numeric_id = re.sub(r"[^0-9]", "", paper_id)
                    if numeric_id:
                        result["abstracts"][numeric_id] = abstract

    # Load jel.csv
    jel_path = metadata_path / "jel.csv"
    if jel_path.exists():
        jel_df = pd.read_csv(jel_path)
        for _, row in jel_df.iterrows():
            paper_id = str(row.get("paper", ""))
            jel_code = str(row.get("jel", ""))
            numeric_id = re.sub(r"[^0-9]", "", paper_id)
            if numeric_id:
                if numeric_id not in result["jel_codes"]:
                    result["jel_codes"][numeric_id] = []
                result["jel_codes"][numeric_id].append(jel_code)

    # Load published.csv (format: "w####\tPublication info...")
    # Note: Variable number of fields due to commas in publication info.
    pub_path = metadata_path / "published.csv"
    if pub_path.exists():
        with open(pub_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header row
            for row in reader:
                full_text = ",".join(row)
                if "\t" in full_text:
                    paper_id, pub_info = full_text.split("\t", 1)
                    numeric_id = re.sub(r"[^0-9]", "", paper_id)
                    if numeric_id:
                        result["published"][numeric_id] = pub_info

    # Load ref.csv
    # Note: Some rows have unescaped quotes in title fields (e.g., 'Is Addiction "Rational"?')
    # which pandas treats as bad lines and skips with on_bad_lines='skip'.
    # Use csv module which is more lenient with non-standard CSV formatting.
    ref_path = metadata_path / "ref.csv"
    if ref_path.exists():
        with open(ref_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, quotechar='"', doublequote=True)
            header = next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 5:
                    paper_id = row[0]
                    numeric_id = re.sub(r"[^0-9]", "", paper_id)
                    if numeric_id:
                        if numeric_id not in result["references"]:
                            result["references"][numeric_id] = []
                        result["references"][numeric_id].append({
                            "author": row[1],
                            "title": row[2],
                            "issue_date": row[3],
                            "doi": row[4],
                        })

    return result


def parse_published_info(pub_info: str) -> dict:
    """
    Parse publication info string to extract title, author, year, venue.

    This is a heuristic parser and may need adjustment based on actual data format.
    """
    # Default values
    result = {
        "title": "",
        "author": "",
        "publication_year": "",
        "published_venue": "",
    }

    if not pub_info:
        return result

    # Try to extract information (format varies, this is a basic attempt)
    parts = pub_info.split(",")
    if len(parts) >= 1:
        result["published_venue"] = parts[0].strip()
    if len(parts) >= 2:
        # Try to find year
        for part in parts:
            year_match = re.search(r"(19|20)\d{2}", part)
            if year_match:
                result["publication_year"] = year_match.group()
                break

    return result


def run_step5(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    nber_metadata_dir: str = DEFAULT_NBER_METADATA_DIR,
    pdf_dir: str = DEFAULT_PDF_DIR,
    output_filename: str = "final_results.xlsx",
    step1_results_path: Optional[str] = None,
    step2_results_path: Optional[str] = None,
    step3_results_path: Optional[str] = None,
    step4_results_path: Optional[str] = None,
    # Filtering parameters for step4 results
    filter_min_subscore: float = 2.0,
    filter_min_score_sum: float = 15,
    apply_filter: bool = True,
) -> str:
    """
    Run Step 5: Generate final Excel results file.

    Args:
        output_dir: Directory containing step results
        nber_metadata_dir: Directory containing NBER metadata CSVs
        pdf_dir: Directory containing journal subfolders with PDFs (for venue mapping)
        output_filename: Name of output Excel file
        step1_results_path: Path to step1_combined.json
        step2_results_path: Path to step2_combined.json
        step3_results_path: Path to step3_combined.json
        step4_results_path: Path to step4_combined.json
        filter_min_subscore: Minimum threshold for any sub-score. Triplets with
            any sub-score < this value are excluded. Default: 2.0
        filter_min_score_sum: Minimum threshold for score_sum. Triplets with
            score_sum < this value are excluded. Default: 12.0 (for 6 criteria)
        apply_filter: Whether to apply the score-based filtering. Default: True

    Returns:
        Path to generated Excel file
    """
    logger = setup_logging()
    logger.info("Starting Step 5: Generate final results")

    output_path = Path(output_dir)

    # Load step results
    step1_path = step1_results_path or (output_path / "step1_combined.json")
    step2_path = step2_results_path or (output_path / "step2_combined.json")
    step3_path = step3_results_path or (output_path / "step3_combined.json")
    step4_path = step4_results_path or (output_path / "step4_combined.json")

    step1_data = load_json(step1_path)
    step2_data = load_json(step2_path)
    step3_data = load_json(step3_path)
    step4_data = load_json(step4_path)

    step1_results = step1_data.get("results", {})
    step2_results = step2_data.get("results", {})
    step3_results = step3_data.get("results", {})
    step4_results = step4_data.get("results", {})

    # Load NBER metadata
    logger.info(f"Loading NBER metadata from {nber_metadata_dir}")
    nber_metadata = load_nber_metadata(nber_metadata_dir)

    # Build paper_id -> venue mapping from PDF folder structure
    logger.info(f"Building venue mapping from {pdf_dir}")
    venue_mapping = build_paper_venue_mapping(pdf_dir)
    logger.info(f"Found {len(venue_mapping)} paper-venue mappings")

    # Helper function to get paper metadata
    def get_paper_metadata(paper_id: str) -> dict:
        """Get metadata for a paper from NBER metadata or references."""
        ref_list = nber_metadata["references"].get(paper_id, [])

        # Get venue from PDF folder structure
        published_venue = venue_mapping.get(paper_id, "")

        # Try to get from references first (usually has author and title)
        if ref_list:
            first_ref = ref_list[0]
            return {
                "title": first_ref.get("title", ""),
                "author": first_ref.get("author", ""),
                "publication_year": str(first_ref.get("issue_date", ""))[:4] if first_ref.get("issue_date") else "",
                "published_venue": published_venue,
            }

        # Fallback with venue from folder
        return {
            "title": "",
            "author": "",
            "publication_year": "",
            "published_venue": published_venue,
        }

    # Create step1 sheet data
    step1_rows = []
    for paper_id, triplets in step1_results.items():
        paper_meta = get_paper_metadata(paper_id)
        paper_url = f"https://www.nber.org/papers/w{paper_id}"

        jel_codes = nber_metadata["jel_codes"].get(paper_id, [])
        for triplet in triplets:
            evidence = triplet.get("supporting_evidence", [])
            row = {
                "title": paper_meta["title"],
                "author": paper_meta["author"],
                "publication_year": paper_meta["publication_year"],
                "published_venue": paper_meta["published_venue"],
                "paper_id": paper_id,
                "jel_codes": ", ".join(jel_codes) if jel_codes else "",
                "treatment": triplet.get("treatment", ""),
                "outcome": triplet.get("outcome", ""),
                "sign": clean_sign(triplet.get("sign", "")),
                "evidence_1": evidence[0] if len(evidence) > 0 else "",
                "evidence_2": evidence[1] if len(evidence) > 1 else "",
                "evidence_3": evidence[2] if len(evidence) > 2 else "",
                "paper_url": paper_url,
            }
            step1_rows.append(row)

    # Create step2 sheet data
    step2_rows = []
    for paper_id, metadata in step2_results.items():
        paper_meta = get_paper_metadata(paper_id)
        paper_url = f"https://www.nber.org/papers/w{paper_id}"
        jel_codes = nber_metadata["jel_codes"].get(paper_id, [])

        id_methods = metadata.get("identification_methods", [])
        row = {
            "title": paper_meta["title"],
            "author": paper_meta["author"],
            "publication_year": paper_meta["publication_year"],
            "published_venue": paper_meta["published_venue"],
            "paper_id": paper_id,
            "jel_codes": ", ".join(jel_codes) if jel_codes else "",
            "paper_type": metadata.get("paper_metadata", {}).get("paper_type", ""),
            "identification_methods": "; ".join(id_methods) if id_methods else "",
            "context": metadata.get("context", ""),
            "paper_url": paper_url,
        }
        step2_rows.append(row)

    # Create step3 sheet data
    step3_rows = []
    for paper_id, triplets in step3_results.items():
        paper_meta = get_paper_metadata(paper_id)
        paper_url = f"https://www.nber.org/papers/w{paper_id}"
        jel_codes = nber_metadata["jel_codes"].get(paper_id, [])

        for triplet in triplets:
            evidence = triplet.get("supporting_evidence", [])
            selection = triplet.get("selection", {})
            context_selected = selection.get("context_selected", [])
            id_method_selected = selection.get("id_method_selected", [])

            row = {
                "title": paper_meta["title"],
                "author": paper_meta["author"],
                "publication_year": paper_meta["publication_year"],
                "published_venue": paper_meta["published_venue"],
                "paper_id": paper_id,
                "jel_codes": ", ".join(jel_codes) if jel_codes else "",
                "treatment": triplet.get("treatment", ""),
                "outcome": triplet.get("outcome", ""),
                "sign": clean_sign(triplet.get("sign", "")),
                "evidence_1": evidence[0] if len(evidence) > 0 else "",
                "evidence_2": evidence[1] if len(evidence) > 1 else "",
                "evidence_3": evidence[2] if len(evidence) > 2 else "",
                "selected_context": "; ".join(context_selected) if context_selected else "",
                "selected_id_method": "; ".join(id_method_selected) if id_method_selected else "",
                "paper_url": paper_url,
            }
            step3_rows.append(row)

    # Create step4 sheet data
    # Check if results are from multi-LLM mode and if context evaluation is enabled
    is_multi_llm = step4_data.get("scoring_mode") == "multi_llm"
    # Auto-detect context evaluation by checking if any triplet has context_appropriateness score
    with_context_eval = step4_data.get("with_context_eval", False)
    if not with_context_eval:
        # Auto-detect from scores in results
        for paper_id, triplets in step4_results.items():
            for triplet in triplets:
                scores = triplet.get("scores", {})
                if scores and "context_appropriateness" in scores:
                    with_context_eval = True
                    break
            if with_context_eval:
                break
    llm_models = ["gemini", "grok", "qwen"] if is_multi_llm else []

    # Helper function to check if triplet passes score filter
    def passes_score_filter(triplet: dict) -> bool:
        """
        Check if triplet passes the score filter criteria.
        Returns True if triplet should be INCLUDED, False if it should be EXCLUDED.

        Filter criteria (when apply_filter=True):
        - Exclude if any sub-score < filter_min_subscore
        - Exclude if score_sum < filter_min_score_sum
        """
        if not apply_filter:
            return True

        scores = triplet.get("scores", {})
        if not scores:
            return False  # No scores means cannot pass filter

        # Get all sub-scores
        sub_score_keys = ["variable_extraction", "direction", "sign", "causality", "main_claim"]
        if with_context_eval:
            sub_score_keys.append("context_appropriateness")
        sub_scores = []
        for key in sub_score_keys:
            val = scores.get(key)
            if val is not None and val != "":
                try:
                    sub_scores.append(float(val))
                except (ValueError, TypeError):
                    pass

        if not sub_scores:
            return False  # No valid scores

        # Check if any sub-score < threshold
        if any(s < filter_min_subscore for s in sub_scores):
            return False

        # Calculate and check score_sum
        score_sum = sum(sub_scores)
        if score_sum < filter_min_score_sum:
            return False

        return True

    # Track filtering statistics
    total_triplets_before_filter = 0
    filtered_out_triplets = 0

    step4_rows = []
    for paper_id, triplets in step4_results.items():
        paper_meta = get_paper_metadata(paper_id)
        paper_url = f"https://www.nber.org/papers/w{paper_id}"
        jel_codes = nber_metadata["jel_codes"].get(paper_id, [])

        # Get global context from step2 for fallback
        step2_metadata = step2_results.get(paper_id, {})
        global_context = step2_metadata.get("context", "")
        global_id_methods = step2_metadata.get("identification_methods", [])

        for triplet in triplets:
            total_triplets_before_filter += 1

            # Apply score filter
            if not passes_score_filter(triplet):
                filtered_out_triplets += 1
                continue

            evidence = triplet.get("supporting_evidence", [])
            selection = triplet.get("selection", {})

            # Handle both multi-LLM and single-model results
            if is_multi_llm:
                # Multi-LLM mode: scores are already averaged
                scores = triplet.get("scores", {})
                per_model_reasons = triplet.get("per_model_reasons", {})
            else:
                # Single-model mode (legacy)
                scores = triplet.get("scores", {})
                per_model_reasons = {}

            # Determine final context and id_methods
            context_selected = selection.get("context_selected", [])
            id_method_selected = selection.get("id_method_selected", [])

            final_context = "; ".join(context_selected) if context_selected else global_context
            final_id_methods = "; ".join(id_method_selected) if id_method_selected else "; ".join(global_id_methods)

            # Calculate score_sum from averaged scores
            score_sum_components = [
                scores.get("variable_extraction", 0),
                scores.get("direction", 0),
                scores.get("sign", 0),
                scores.get("causality", 0),
                scores.get("main_claim", 0),
            ]
            if with_context_eval:
                score_sum_components.append(scores.get("context_appropriateness", 0))
            score_sum = sum(score_sum_components) if scores else 0

            # Build base row
            row = {
                "title": paper_meta["title"],
                "author": paper_meta["author"],
                "publication_year": paper_meta["publication_year"],
                "published_venue": paper_meta["published_venue"],
                "paper_id": paper_id,
                "jel_codes": ", ".join(jel_codes) if jel_codes else "",
                "treatment": triplet.get("treatment", ""),
                "outcome": triplet.get("outcome", ""),
                "sign": clean_sign(triplet.get("sign", "")),
                "evidence_1": evidence[0] if len(evidence) > 0 else "",
                "evidence_2": evidence[1] if len(evidence) > 1 else "",
                "evidence_3": evidence[2] if len(evidence) > 2 else "",
                "final_context": final_context,
                "final_id_methods": final_id_methods,
                # Averaged scores (from 3 LLMs in multi-LLM mode)
                "variable_extraction": scores.get("variable_extraction", "") if scores else "",
                "direction": scores.get("direction", "") if scores else "",
                "sign_score": scores.get("sign", "") if scores else "",
                "causality": scores.get("causality", "") if scores else "",
                "main_claim": scores.get("main_claim", "") if scores else "",
            }
            # Add context_appropriateness if enabled
            if with_context_eval:
                row["context_appropriateness"] = scores.get("context_appropriateness", "") if scores else ""
            row["score_sum"] = score_sum if scores else ""
            row["paper_url"] = paper_url

            # Add per-LLM reason columns after paper_url
            if is_multi_llm:
                reason_types = ["variable_extraction", "direction", "sign", "causality", "main_claim"]
                if with_context_eval:
                    reason_types.append("context_appropriateness")
                for reason_type in reason_types:
                    for model in llm_models:
                        col_name = f"reason_{reason_type}_{model}"
                        model_reasons = per_model_reasons.get(model, {})
                        row[col_name] = model_reasons.get(reason_type, "") if model_reasons else ""
            else:
                # Single-model mode: use legacy reasons field
                reasons = triplet.get("reasons", {})
                row["reason_variable_extraction"] = reasons.get("variable_extraction", "") if reasons else ""
                row["reason_direction"] = reasons.get("direction", "") if reasons else ""
                row["reason_sign"] = reasons.get("sign", "") if reasons else ""
                row["reason_causality"] = reasons.get("causality", "") if reasons else ""
                row["reason_main_claim"] = reasons.get("main_claim", "") if reasons else ""
                if with_context_eval:
                    row["reason_context_appropriateness"] = reasons.get("context_appropriateness", "") if reasons else ""

            step4_rows.append(row)

    # Create DataFrames
    step1_df = pd.DataFrame(step1_rows) if step1_rows else pd.DataFrame()
    step2_df = pd.DataFrame(step2_rows) if step2_rows else pd.DataFrame()
    step3_df = pd.DataFrame(step3_rows) if step3_rows else pd.DataFrame()
    step4_df = pd.DataFrame(step4_rows) if step4_rows else pd.DataFrame()

    # Sort all DataFrames by paper_id (ascending, numeric order)
    for df in [step1_df, step2_df, step3_df, step4_df]:
        if not df.empty and "paper_id" in df.columns:
            df["_paper_id_num"] = pd.to_numeric(df["paper_id"], errors="coerce")
            df.sort_values("_paper_id_num", ascending=True, inplace=True)
            df.drop(columns=["_paper_id_num"], inplace=True)
            df.reset_index(drop=True, inplace=True)

    # Reorder step4_df columns to ensure paper_url comes before reason columns
    if not step4_df.empty:
        base_cols = [
            "title", "author", "publication_year", "published_venue", "paper_id", "jel_codes",
            "treatment", "outcome", "sign",
            "evidence_1", "evidence_2", "evidence_3",
            "final_context", "final_id_methods",
            "variable_extraction", "direction", "sign_score", "causality", "main_claim",
        ]
        # Add context_appropriateness before score_sum if enabled
        if with_context_eval:
            base_cols.append("context_appropriateness")
        base_cols.extend(["score_sum", "paper_url"])

        # Define reason column order for multi-LLM mode
        if is_multi_llm:
            reason_types = ["variable_extraction", "direction", "sign", "causality", "main_claim"]
            if with_context_eval:
                reason_types.append("context_appropriateness")
            reason_cols = []
            for reason_type in reason_types:
                for model in llm_models:
                    col_name = f"reason_{reason_type}_{model}"
                    if col_name in step4_df.columns:
                        reason_cols.append(col_name)
        else:
            # Single-model mode: legacy reason columns
            reason_cols = [
                "reason_variable_extraction", "reason_direction", "reason_sign",
                "reason_causality", "reason_main_claim"
            ]
            if with_context_eval:
                reason_cols.append("reason_context_appropriateness")
            reason_cols = [c for c in reason_cols if c in step4_df.columns]

        # Reorder: base_cols first, then reason columns
        ordered_cols = [c for c in base_cols if c in step4_df.columns] + reason_cols
        step4_df = step4_df[ordered_cols]

    # Create readme sheet
    readme_data = {
        "Sheet": ["step1", "step2", "step3", "step4"],
        "Description": [
            "Extracted causal relations (treatment, outcome, sign, evidence)",
            "Paper metadata (paper_type, identification_methods, context)",
            "Selected context and identification methods for each triplet",
            "Critic evaluation scores and reasons for each triplet",
        ],
        "Rows": [
            len(step1_rows),
            len(step2_rows),
            len(step3_rows),
            len(step4_rows),
        ],
    }
    readme_df = pd.DataFrame(readme_data)

    # Write to Excel (clean illegal characters first)
    output_file = output_path / output_filename
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        clean_for_excel(readme_df).to_excel(writer, sheet_name="readme", index=False)
        if not step1_df.empty:
            clean_for_excel(step1_df).to_excel(writer, sheet_name="step1", index=False)
        if not step2_df.empty:
            clean_for_excel(step2_df).to_excel(writer, sheet_name="step2", index=False)
        if not step3_df.empty:
            clean_for_excel(step3_df).to_excel(writer, sheet_name="step3", index=False)
        if not step4_df.empty:
            clean_for_excel(step4_df).to_excel(writer, sheet_name="step4", index=False)

        # Add clickable hyperlinks for paper_url columns
        from openpyxl.styles import Font
        workbook = writer.book
        link_font = Font(color="0563C1", underline="single")

        for sheet_name, df in [
            ("step1", step1_df),
            ("step2", step2_df),
            ("step3", step3_df),
            ("step4", step4_df),
        ]:
            if df.empty or "paper_url" not in df.columns:
                continue
            ws = workbook[sheet_name]
            url_col_idx = list(df.columns).index("paper_url") + 1  # 1-indexed
            for row_idx in range(2, len(df) + 2):  # Start from row 2 (skip header)
                cell = ws.cell(row=row_idx, column=url_col_idx)
                if cell.value and str(cell.value).startswith("http"):
                    cell.hyperlink = str(cell.value)
                    cell.font = link_font

    logger.info(f"Step 5 complete. Results saved to {output_file}")
    logger.info(f"  - step1: {len(step1_rows)} triplets")
    logger.info(f"  - step2: {len(step2_rows)} papers")
    logger.info(f"  - step3: {len(step3_rows)} triplets")
    logger.info(f"  - step4: {len(step4_rows)} triplets")
    if apply_filter:
        logger.info(f"  - step4 filtering: {filtered_out_triplets}/{total_triplets_before_filter} excluded "
                    f"(min_subscore>={filter_min_subscore}, min_score_sum>={filter_min_score_sum})")

    return str(output_file)


def main():
    """Command-line interface for Step 5."""
    parser = argparse.ArgumentParser(
        description="Step 5: Generate final Excel results file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing step results"
    )
    parser.add_argument(
        "--nber-metadata-dir",
        type=str,
        default=DEFAULT_NBER_METADATA_DIR,
        help="Directory containing NBER metadata CSVs"
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=DEFAULT_PDF_DIR,
        help="Directory containing journal subfolders with PDFs (for venue mapping)"
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="final_results.xlsx",
        help="Name of output Excel file"
    )
    parser.add_argument(
        "--step1-results",
        type=str,
        help="Path to step1_combined.json"
    )
    parser.add_argument(
        "--step2-results",
        type=str,
        help="Path to step2_combined.json"
    )
    parser.add_argument(
        "--step3-results",
        type=str,
        help="Path to step3_combined.json"
    )
    parser.add_argument(
        "--step4-results",
        type=str,
        help="Path to step4_combined.json"
    )
    # Filtering parameters
    parser.add_argument(
        "--filter-min-subscore",
        type=float,
        default=2.0,
        help="Minimum threshold for sub-scores. Triplets with any sub-score < this are excluded. Default: 2.0"
    )
    parser.add_argument(
        "--filter-min-score-sum",
        type=float,
        default=15.0,
        help="Minimum threshold for score_sum. Triplets with score_sum < this are excluded. Default: 12.0 (for 6 criteria)"
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable score-based filtering"
    )

    args = parser.parse_args()

    run_step5(
        output_dir=args.output_dir,
        nber_metadata_dir=args.nber_metadata_dir,
        pdf_dir=args.pdf_dir,
        output_filename=args.output_filename,
        step1_results_path=args.step1_results,
        step2_results_path=args.step2_results,
        step3_results_path=args.step3_results,
        step4_results_path=args.step4_results,
        filter_min_subscore=args.filter_min_subscore,
        filter_min_score_sum=args.filter_min_score_sum,
        apply_filter=not args.no_filter,
    )


if __name__ == "__main__":
    main()
