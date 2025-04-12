import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from app.config import MODEL_NAME, HF_TOKEN, PROMPT_TEMPLATE, PRIMARY_MEDICATIONS, SECONDARY_MEDICATIONS, \
    GENERATION_PARAMS


class InsomniaClassifier:
    def __init__(self):
        """Initialize the InsomniaClassifier with model, tokenizer, and medication lists."""
        # Load medication dictionaries
        self.primary_medications = PRIMARY_MEDICATIONS
        self.secondary_medications = SECONDARY_MEDICATIONS

        # 8-bit quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True  # Enables 8-bit quantization
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

        # Load model with 8-bit quantization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,  # Apply 8-bit quantization
            device_map="auto",  # Automatically assigns to available devices
            token=HF_TOKEN
        )

        print(f"Model loaded on {self.device}")

    def classify(self, text):
        """
        Classifies text, extracts relevant phrases, and determines 'yes' or 'no' labels.

        Args:
            text (str): Clinical text to analyze

        Returns:
            tuple: (classification_results, extracted_text_snippets)
        """
        results = {}  # Store classification labels (yes/no)
        extracted_text = {
            "Definition 1 Extracted": "", "Definition 2 Extracted": "",
            "Rule A Extracted": "", "Rule B Extracted": "", "Rule C Extracted": ""
        }

        # Get response from model
        combined_prompt = PROMPT_TEMPLATE.format(text=text)
        combined_response = self._evaluate_with_model(combined_prompt)

        # Parse the combined response to extract separate results
        sleep_difficulty = ""
        daytime_impairment = ""

        # Clean up response
        combined_response = re.sub(r".*<start_of_turn>model*", "", combined_response, flags=re.DOTALL).strip()

        # Extract sleep difficulty phrases
        if "Sleep Difficulty Phrases:" in combined_response:
            sleep_section = \
            combined_response.split("Sleep Difficulty Phrases:")[1].split("Daytime Impairment Phrases:")[0].strip()
            print(sleep_section)
            if "unknown" in sleep_section.lower() and len(sleep_section.strip()) <= 10:
                sleep_difficulty = ""
            else:
                sleep_difficulty = sleep_section

        # Extract daytime impairment phrases
        if "Daytime Impairment Phrases:" in combined_response:
            daytime_section = combined_response.split("Daytime Impairment Phrases:")[1].strip()
            print(daytime_section)
            if "unknown" in daytime_section.lower() and len(daytime_section.strip()) <= 10:
                daytime_impairment = ""
            else:
                daytime_impairment = daytime_section

        # Store extracted text
        extracted_text["Definition 1 Extracted"] = sleep_difficulty
        extracted_text["Definition 2 Extracted"] = daytime_impairment

        # Rule A: Insomnia Diagnosis (Yes if both Definition 1 & 2 are Yes)
        results["Definition 1 (Sleep Difficulty)"] = "yes" if extracted_text["Definition 1 Extracted"] else "no"
        results["Definition 2 (Daytime Impairment)"] = "yes" if extracted_text["Definition 2 Extracted"] else "no"
        results["Rule A (Insomnia Diagnosis)"] = "yes" if results["Definition 1 (Sleep Difficulty)"] == "yes" and \
                                                          results[
                                                              "Definition 2 (Daytime Impairment)"] == "yes" else "no"

        # Rule B: Primary Insomnia Medications
        extracted_text["Rule B Extracted"] = self.extract_medications(text, self.primary_medications)
        results["Rule B (Primary Medications)"] = "yes" if extracted_text["Rule B Extracted"] else "no"

        # Rule C: Secondary Insomnia Medications
        extracted_text["Rule C Extracted"] = self.extract_medications(text, self.secondary_medications)
        results["Rule C (Secondary Medications)"] = "yes" if extracted_text["Rule C Extracted"] and (
                    results["Definition 1 (Sleep Difficulty)"] == "yes" or results[
                "Definition 2 (Daytime Impairment)"] == "yes") else "no"

        # If Rule C is 'no', empty its extracted text
        if results["Rule C (Secondary Medications)"] == "no":
            extracted_text["Rule C Extracted"] = ""

        # Final Insomnia Status: Yes if any Rule is Yes
        results["Final Insomnia Status"] = "yes" if any([
            results["Rule A (Insomnia Diagnosis)"] == "yes",
            results["Rule B (Primary Medications)"] == "yes",
            results["Rule C (Secondary Medications)"] == "yes"
        ]) else "no"

        return results, extracted_text

    def _evaluate_with_model(self, prompt):
        """
        Process prompt with the loaded model.

        Args:
            prompt (str): Input prompt to send to the model

        Returns:
            str: Model response
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                temperature=GENERATION_PARAMS["temperature"],
                max_new_tokens=GENERATION_PARAMS["max_new_tokens"],
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Extract only the generated text (not the input prompt)
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Clean up response
        response = response.strip()
        response = response.split("<end_of_turn>")[0].strip() if "<end_of_turn>" in response else response
        print("Model Response:\n", response)
        print("#" * 40)

        return response

    def extract_medications(self, text, medication_list):
        """
        Extracts medication names from the text based on given list.

        Args:
            text (str): Text to search for medications
            medication_list (dict): Dictionary of medications to look for

        Returns:
            str: Comma-separated list of found medications
        """
        extracted_meds = [med for med in medication_list if re.search(rf"\b{med}\b", text, re.IGNORECASE)]
        return ", ".join(extracted_meds)