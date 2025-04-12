import os
import torch
import pandas as pd
from app.data_loader import load_data
from app.model import InsomniaClassifier
from app.config import MODEL_NAME, HF_TOKEN, DATASET_PATH, OUTPUT_PATH
from app.output_handler import convert_output_to_json


def main():
    # Set environment variable
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Load validation data
    df_val = load_data(DATASET_PATH)
    texts = df_val['text'].tolist()

    # Initialize classifier
    classifier = InsomniaClassifier()

    classification_results = []
    extracted_texts = []

    # Process each clinical note with exception handling
    for idx, text in enumerate(texts):
        print(f"Processing text {idx + 1}/{len(texts)}: {text[:100]}...")
        try:
            classification, extracted = classifier.classify(text)
            classification_results.append(classification)
            extracted_texts.append(extracted)
        except RuntimeError as e:
            print(f"RuntimeError for text at index {idx}: {e}")
            # Append default classifications and empty extracted text on error
            classification_results.append({
                "Definition 1 (Sleep Difficulty)": "no",
                "Definition 2 (Daytime Impairment)": "no",
                "Rule A (Insomnia Diagnosis)": "no",
                "Rule B (Primary Medications)": "no",
                "Rule C (Secondary Medications)": "no",
                "Final Insomnia Status": "no"
            })
            extracted_texts.append({
                "Definition 1 Extracted": "",
                "Definition 2 Extracted": "",
                "Rule A Extracted": "",
                "Rule B Extracted": "",
                "Rule C Extracted": ""
            })
        print("-" * 80)

    # Convert results to DataFrames
    df_classification = pd.DataFrame(classification_results)
    df_extracted = pd.DataFrame(extracted_texts)

    # Combine all DataFrames
    df_final = pd.concat([df_val[['text', 'note_id']], df_classification, df_extracted], axis=1)

    # Rename columns to match expected names for JSON conversion
    df_final = df_final.rename(columns={
        "Definition 1 (Sleep Difficulty)": "Definition 1 Pred",
        "Definition 2 (Daytime Impairment)": "Definition 2 Pred",
        "Rule A (Insomnia Diagnosis)": "Rule A Pred",
        "Rule B (Primary Medications)": "Rule B Pred",
        "Rule C (Secondary Medications)": "Rule C Pred",
        "Final Insomnia Status": "Insomnia Pred",
        "Definition 1 Extracted": "Definition 1 Evidence",
        "Definition 2 Extracted": "Definition 2 Evidence",
        "Rule B Extracted": "Rule B Evidence",
        "Rule C Extracted": "Rule C Evidence"
    })

    # Save CSV to results folder
    csv_output_path = os.path.join("results", OUTPUT_PATH)
    df_final.to_csv(csv_output_path, index=False)

    # Generate JSON outputs
    convert_output_to_json(csv_output_path)

    return df_final


if __name__ == "__main__":
    main()