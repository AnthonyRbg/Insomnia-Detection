# Model configuration
MODEL_NAME = "google/gemma-2-2b-it"
HF_TOKEN = "<your_hugging_face_token>"

# Path configurations
DATASET_PATH = "data/testing.csv"
OUTPUT_PATH = "output.csv"

# Model generation parameters
GENERATION_PARAMS = {
    "temperature": 0.0,
    "max_new_tokens": 512,
}
# Prompt template for insomnia classification
PROMPT_TEMPLATE = """<start_of_turn>user
You are a clinical NLP annotator with expertise in identifying insomnia-related symptoms. Your task is to extract two specific types of information from the following clinical note: one related to nighttime sleep difficulty and the other to daytime impairment due to poor sleep.

Follow the instructions carefully. Extract only the exact phrases that directly support each category. Do not extract irrelevant clinical content, general symptoms, or any non-sleep-related information.

---

###Type 1: Sleep Difficulty (Nighttime)
Identify phrases that describe difficulty sleeping during the night. This includes situations where the patient reports trouble falling asleep (such as lying awake for long periods before sleeping), waking up multiple times throughout the night and struggling to stay asleep, waking up much earlier than intended and being unable to return to sleep, or explicitly using the word "insomnia" or phrases like "unable to sleep." These must clearly refer to **nighttime** sleep challenges.

You must **not** extract general phrases like "feels tired" or "has low energy," nor should you extract daytime consequences or symptoms not related to sleep itself. Avoid pulling in any unrelated medical content like pain, lab results, infections, or nausea.

---

###Type 2: Daytime Impairment (Due to Poor Sleep)
Identify phrases that reflect the patient's impaired ability to function during the day as a result of poor sleep. These may include feeling excessively fatigued or unwell during the day, trouble focusing, forgetfulness, reduced motivation or drive, emotional instability (such as irritability or mood swings), behavioral issues like impulsivity or hyperactivity, or even statements showing dissatisfaction with sleep quality. Phrases like "can't concentrate," "daytime sleepiness," or "complains about sleep quality" are relevant here — but only if they clearly reflect **daytime effects**.

Do **not** include any nighttime complaints like "couldn't sleep last night," and do not extract other unrelated clinical symptoms or treatments unless they are directly tied to daytime effects from poor sleep.

---

### General Rules:
- Extract only the **exact phrases** from the clinical note that match the descriptions above.
- Do NOT paraphrase, summarize, or explain anything.
- Do NOT extract any content about unrelated medical conditions or treatments.
- If no relevant phrase is found for a category, return only `"unknown"` (in quotes).
- Your output MUST follow the format shown below, with two clearly separated sections.

---

### Clinical Note:
{text}

---

### ✅ Response Format:
Sleep Difficulty Phrases:
[exact extracted phrases for nighttime sleep difficulty OR "unknown"]

Daytime Impairment Phrases:
[exact extracted phrases for daytime impairment OR "unknown"]
<end_of_turn>
<start_of_turn>model
"""

# Medication dictionaries
PRIMARY_MEDICATIONS = {
    # Generic names
    "Estazolam": "Estazolam",
    "Eszopiclone": "Eszopiclone",
    "Flurazepam": "Flurazepam",
    "Lemborexant": "Lemborexant",
    "Quazepam": "Quazepam",
    "Ramelteon": "Ramelteon",
    "Suvorexant": "Suvorexant",
    "Temazepam": "Temazepam",
    "Triazolam": "Triazolam",
    "Zaleplon": "Zaleplon",
    "Zolpidem": "Zolpidem",

    # Brand names mapped to generic
    "Ambien": "Zolpidem",
    "Ambien CR": "Zolpidem",
    "Edluar": "Zolpidem",
    "Intermezzo": "Zolpidem",
    "Sonata": "Zaleplon",
    "Doral": "Quazepam",
    "Lunesta": "Eszopiclone",
    "Rozerem": "Ramelteon",
    "Belsomra": "Suvorexant",
    "Dayvigo": "Lemborexant",
    "Halcion": "Triazolam"
}

SECONDARY_MEDICATIONS = {
    # Generic names
    "Acamprosate": "Acamprosate",
    "Alprazolam": "Alprazolam",
    "Clonazepam": "Clonazepam",
    "Clonidine": "Clonidine",
    "Diazepam": "Diazepam",
    "Diphenhydramine": "Diphenhydramine",
    "Doxepin": "Doxepin",
    "Gabapentin": "Gabapentin",
    "Hydroxyzine": "Hydroxyzine",
    "Lorazepam": "Lorazepam",
    "Melatonin": "Melatonin",
    "Mirtazapine": "Mirtazapine",
    "Olanzapine": "Olanzapine",
    "Quetiapine": "Quetiapine",
    "Trazodone": "Trazodone",
    "Amitriptyline": "Amitriptyline",

    # Brand names mapped to generic
    "Ativan": "Lorazepam",
    "Benadryl": "Diphenhydramine",
    "Restoril": "Temazepam",
    "Seroquel": "Quetiapine",
    "Neurontin": "Gabapentin",
    "Klonopin": "Clonazepam",
    "Valium": "Diazepam",
    "Desyrel": "Trazodone",
    "Elavil": "Amitriptyline",
    "Remeron": "Mirtazapine",
    "Vistaril": "Hydroxyzine",
    "Atarax": "Hydroxyzine",
    "Xanax": "Alprazolam",

    # Common misspellings mapped to generic
    "Trazadone": "Trazodone",
    "Clonazapam": "Clonazepam",
    "Seraquil": "Quetiapine"
}

