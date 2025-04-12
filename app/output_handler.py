import os
import pandas as pd
import json


def convert_output_to_json(csv_file_path):
    """
    Convert CSV output to three different JSON formats and save them in the results folder.

    Args:
        csv_file_path (str): Path to the CSV file with classification results
    """
    # Load the CSV file
    df = pd.read_csv(csv_file_path, dtype=str)

    # Clean column names (remove leading/trailing whitespace)
    df.columns = df.columns.str.strip()

    # Generate subtask_1.json
    generate_subtask_1(df)

    # Generate subtask_2a.json
    generate_subtask_2a(df)

    # Generate subtask_2b.json
    generate_subtask_2b(df)


def safe_get(value):
    """
    Safely extract and clean cell values.

    Args:
        value: Value from DataFrame cell

    Returns:
        str: Cleaned value or "no" if empty
    """
    return value.strip().lower() if pd.notna(value) and value.strip() else "no"


def process_text(text):
    """
    Convert empty or NaN values to an empty list.

    Args:
        text: Text evidence from DataFrame

    Returns:
        list: List containing the text or empty list if no text
    """
    return [text] if pd.notna(text) and text.strip() else []


def generate_subtask_1(df):
    """
    Generate subtask_1.json with overall insomnia classification.

    Args:
        df (pandas.DataFrame): DataFrame with classification results
    """
    result = {}

    for _, row in df.iterrows():
        note_id = str(row["note_id"]).strip()  # Ensure it's a string and trimmed

        result[note_id] = {
            "Insomnia": safe_get(row.get("Insomnia Pred", "NA")),
        }

    # Save to JSON file
    json_file = os.path.join("results", "subtask_1.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"JSON file saved at {json_file}")


def generate_subtask_2a(df):
    """
    Generate subtask_2a.json with detailed classification labels.

    Args:
        df (pandas.DataFrame): DataFrame with classification results
    """
    result = {}

    for _, row in df.iterrows():
        note_id = str(row["note_id"]).strip()  # Ensure it's a string and trimmed

        result[note_id] = {
            "Definition 1": safe_get(row.get("Definition 1 Pred", "NA")),
            "Definition 2": safe_get(row.get("Definition 2 Pred", "NA")),
            "Rule A": safe_get(row.get("Rule A Pred", "NA")),
            "Rule B": safe_get(row.get("Rule B Pred", "NA")),
            "Rule C": safe_get(row.get("Rule C Pred", "NA")),
        }

    # Save to JSON file
    json_file = os.path.join("results", "subtask_2a.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"JSON file saved at {json_file}")


def generate_subtask_2b(df):
    """
    Generate subtask_2b.json with classification labels and evidence text.

    Args:
        df (pandas.DataFrame): DataFrame with classification results
    """
    result = {}

    for _, row in df.iterrows():
        note_id = str(row["note_id"])  # Ensure Note_id is treated as a string

        result[note_id] = {
            "Definition 1": {
                "label": safe_get(row.get("Definition 1 Pred", "NA")),
                "text": process_text(row.get("Definition 1 Evidence", "NA"))
            },
            "Definition 2": {
                "label": safe_get(row.get("Definition 2 Pred", "NA")),
                "text": process_text(row.get("Definition 2 Evidence", "NA"))
            },
            "Rule B": {
                "label": safe_get(row.get("Rule B Pred", "NA")),
                "text": process_text(row.get("Rule B Evidence", "NA"))
            },
            "Rule C": {
                "label": safe_get(row.get("Rule C Pred", "NA")),
                "text": process_text(row.get("Rule C Evidence", "NA"))
            }
        }

    # Save to JSON file
    json_file = os.path.join("results", "subtask_2b.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"JSON file saved at {json_file}")