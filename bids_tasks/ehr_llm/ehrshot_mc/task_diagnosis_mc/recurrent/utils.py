# Define condition dictionary for efficient lookup by condition_name
CONDITIONS_DICT = {
    "hypertension": {"condition_concept_name": "Essential hypertension", "condition_concept_id": 320128},
    "obesity": {"condition_concept_name": "Obesity", "condition_concept_id": 433736},
    "asthma": {"condition_concept_name": "Asthma", "condition_concept_id": 317009},
    "hyperlipidemia": {"condition_concept_name": "Hyperlipidemia", "condition_concept_id": 432867},
    "depressive_disorder": {"condition_concept_name": "Depressive disorder", "condition_concept_id": 440383},
    "chronic_kidney_disease": {"condition_concept_name": "Chronic kidney disease", "condition_concept_id": 46271022},
    "chronic_obstructive_pulmonary_disease": {"condition_concept_name": "Chronic obstructive pulmonary disease", "condition_concept_id": 255573},
    "coronary_arteriosclerosis": {"condition_concept_name": "Coronary arteriosclerosis", "condition_concept_id": 317576},
    "type_ii_diabetes": {"condition_concept_name": "Type 2 diabetes mellitus", "condition_concept_id": 201826},
    "breast_cancer": {"condition_concept_name": "Malignant tumor of breast", "condition_concept_id": 4112853},
    "prostate_cancer": {"condition_concept_name": "Malignant tumor of prostate", "condition_concept_id": 4163261},
    "pancreatic_cancer": {"condition_concept_name": "Malignant tumor of pancreas", "condition_concept_id": 4180793},
    "lupus": {"condition_concept_name": "Systemic lupus erythematosus", "condition_concept_id": 257628},
    "acute_myocardial_infarction": {"condition_concept_name": "Acute myocardial infarction", "condition_concept_id": 312327},
    "ischemic_stroke": {"condition_concept_name": "Cerebral infarction", "condition_concept_id": 443454},
    "heart_failure": {"condition_concept_name": "Heart failure", "condition_concept_id": 316139},
    "dementia": {"condition_concept_name": "Dementia", "condition_concept_id": 4182210},
    "intestinal_cancer": {"condition_concept_name": "Malignant tumor of intestine", "condition_concept_id": 443398},
    "gastric_cancer": {"condition_concept_name": "Malignant tumor of stomach", "condition_concept_id": 443387},
    "lung_cancer": {"condition_concept_name": "Malignant tumor of lung", "condition_concept_id": 443388},
    "liver_cancer": {"condition_concept_name": "Malignant tumor of liver", "condition_concept_id": 4246127}
}

# Dataset filtering function for RECURRENT diagnosis cases only
def filter_recurrent_cases(dataset, condition_name: str = "hypertension"):
    """Filter dataset to only include 'recurrent' diagnosis cases for subsequent diagnosis prediction"""
    def _filter_recurrent(doc):
        condition_type_key = f"{condition_name}_type"
        return doc.get(condition_type_key) == "recurrent"
    
    # Filter the dataset
    filtered_dataset = dataset.filter(_filter_recurrent)
    print(f"Filtered to {len(filtered_dataset)} recurrent diagnosis cases from {len(dataset)} total")
    
    return filtered_dataset


def doc_to_text(doc, condition_name: str) -> str:
    """
    Generate the input prompt for RECURRENT diagnosis prediction task.
    This function is specifically designed for subsequent diagnosis prediction in patients with prior history.
    
    Args:
        doc: A dictionary containing the patient data with keys:
            - visit_cumulated: Full visit history text
            - {condition_name}_type: Should be "recurrent" (filtered by process_docs)
            - {condition_name}_one_year_diagnosis: Integer (0/1) outcome data
        condition_name: Name of the condition to predict (e.g., "hypertension", "heart_failure")
    
    Returns:
        str: Formatted prompt text for the model focused on RECURRENT diagnosis prediction
    """
    # Extract relevant information from the document
    visit_history = doc["visit_cumulated"]
    
    # Get condition concept name from CONDITIONS_DICT
    condition_info = CONDITIONS_DICT.get(condition_name)
    if condition_info:
        condition_concept_name = condition_info["condition_concept_name"]
    else:
        # Fallback to condition_name if not found
        condition_concept_name = condition_name
    
    # For RECURRENT diagnosis task - this should always be "recurrent" due to filtering
    task_description = f"predicting if the patient will receive a recurrent diagnosis of {condition_concept_name}"
    instruction_text = f"Predict if the patient will receive a recurrent diagnosis of {condition_concept_name} within the next year (we know this patient has had this condition in their medical records up to the current visit)"
    
    # Create the prompt with task description before visit history
    prompt = f"""This is a task about {task_description} within the next year based on their recent EHR visit data: 

{visit_history}

Above are the patient's recent EHR visit records (including the current visit) in chronological order.

You are a medical AI assistant analyzing patient data to predict future diagnoses given the patient's recent and current EHR visit records provided above.

INSTRUCTIONS:
- {instruction_text}
- the condition of {condition_name} means itself and any of its descendant conditions in the medical ontology hierarchy
- Do not provide any explanation, reasoning, or additional text.
- You must respond with exactly one word only: Yes or No

Answer:"""
    
    return prompt


def doc_to_target(doc, condition_name: str) -> int:
    """
    Extract the target label for the diagnosis prediction task.
    
    Args:
        doc: A dictionary containing the patient data with keys:
            - {condition_name}_one_year_diagnosis: Integer (0/1) target label
        condition_name: Name of the condition to predict (e.g., "hypertension", "heart_failure")
    
    Returns:
        int: 0 for "No", 1 for "Yes"
    """
    diagnosis_outcome_key = f"{condition_name}_one_year_diagnosis"
    diagnosis_outcome = doc[diagnosis_outcome_key]
    
    # The outcome is already stored as integer (0/1), so return directly
    return int(diagnosis_outcome)


# Individual condition-specific functions for recurrent diagnosis
# Each condition needs its own function that can be called from YAML

# Hypertension functions
def filter_recurrent_cases_hypertension(dataset):
    return filter_recurrent_cases(dataset, condition_name="hypertension")

def doc_to_text_hypertension(doc):
    return doc_to_text(doc, condition_name="hypertension")

def doc_to_target_hypertension(doc):
    return doc_to_target(doc, condition_name="hypertension")

# Obesity functions  
def filter_recurrent_cases_obesity(dataset):
    return filter_recurrent_cases(dataset, condition_name="obesity")

def doc_to_text_obesity(doc):
    return doc_to_text(doc, condition_name="obesity")

def doc_to_target_obesity(doc):
    return doc_to_target(doc, condition_name="obesity")

# Asthma functions
def filter_recurrent_cases_asthma(dataset):
    return filter_recurrent_cases(dataset, condition_name="asthma")

def doc_to_text_asthma(doc):
    return doc_to_text(doc, condition_name="asthma")

def doc_to_target_asthma(doc):
    return doc_to_target(doc, condition_name="asthma")

# Hyperlipidemia functions
def filter_recurrent_cases_hyperlipidemia(dataset):
    return filter_recurrent_cases(dataset, condition_name="hyperlipidemia")

def doc_to_text_hyperlipidemia(doc):
    return doc_to_text(doc, condition_name="hyperlipidemia")

def doc_to_target_hyperlipidemia(doc):
    return doc_to_target(doc, condition_name="hyperlipidemia")

# Depressive disorder functions
def filter_recurrent_cases_depressive_disorder(dataset):
    return filter_recurrent_cases(dataset, condition_name="depressive_disorder")

def doc_to_text_depressive_disorder(doc):
    return doc_to_text(doc, condition_name="depressive_disorder")

def doc_to_target_depressive_disorder(doc):
    return doc_to_target(doc, condition_name="depressive_disorder")

# Chronic kidney disease functions
def filter_recurrent_cases_chronic_kidney_disease(dataset):
    return filter_recurrent_cases(dataset, condition_name="chronic_kidney_disease")

def doc_to_text_chronic_kidney_disease(doc):
    return doc_to_text(doc, condition_name="chronic_kidney_disease")

def doc_to_target_chronic_kidney_disease(doc):
    return doc_to_target(doc, condition_name="chronic_kidney_disease")

# Chronic obstructive pulmonary disease functions
def filter_recurrent_cases_chronic_obstructive_pulmonary_disease(dataset):
    return filter_recurrent_cases(dataset, condition_name="chronic_obstructive_pulmonary_disease")

def doc_to_text_chronic_obstructive_pulmonary_disease(doc):
    return doc_to_text(doc, condition_name="chronic_obstructive_pulmonary_disease")

def doc_to_target_chronic_obstructive_pulmonary_disease(doc):
    return doc_to_target(doc, condition_name="chronic_obstructive_pulmonary_disease")

# Coronary arteriosclerosis functions
def filter_recurrent_cases_coronary_arteriosclerosis(dataset):
    return filter_recurrent_cases(dataset, condition_name="coronary_arteriosclerosis")

def doc_to_text_coronary_arteriosclerosis(doc):
    return doc_to_text(doc, condition_name="coronary_arteriosclerosis")

def doc_to_target_coronary_arteriosclerosis(doc):
    return doc_to_target(doc, condition_name="coronary_arteriosclerosis")

# Type II diabetes functions
def filter_recurrent_cases_type_ii_diabetes(dataset):
    return filter_recurrent_cases(dataset, condition_name="type_ii_diabetes")

def doc_to_text_type_ii_diabetes(doc):
    return doc_to_text(doc, condition_name="type_ii_diabetes")

def doc_to_target_type_ii_diabetes(doc):
    return doc_to_target(doc, condition_name="type_ii_diabetes")

# Breast cancer functions
def filter_recurrent_cases_breast_cancer(dataset):
    return filter_recurrent_cases(dataset, condition_name="breast_cancer")

def doc_to_text_breast_cancer(doc):
    return doc_to_text(doc, condition_name="breast_cancer")

def doc_to_target_breast_cancer(doc):
    return doc_to_target(doc, condition_name="breast_cancer")

# Prostate cancer functions
def filter_recurrent_cases_prostate_cancer(dataset):
    return filter_recurrent_cases(dataset, condition_name="prostate_cancer")

def doc_to_text_prostate_cancer(doc):
    return doc_to_text(doc, condition_name="prostate_cancer")

def doc_to_target_prostate_cancer(doc):
    return doc_to_target(doc, condition_name="prostate_cancer")

# Pancreatic cancer functions
def filter_recurrent_cases_pancreatic_cancer(dataset):
    return filter_recurrent_cases(dataset, condition_name="pancreatic_cancer")

def doc_to_text_pancreatic_cancer(doc):
    return doc_to_text(doc, condition_name="pancreatic_cancer")

def doc_to_target_pancreatic_cancer(doc):
    return doc_to_target(doc, condition_name="pancreatic_cancer")

# Lupus functions
def filter_recurrent_cases_lupus(dataset):
    return filter_recurrent_cases(dataset, condition_name="lupus")

def doc_to_text_lupus(doc):
    return doc_to_text(doc, condition_name="lupus")

def doc_to_target_lupus(doc):
    return doc_to_target(doc, condition_name="lupus")

# Acute myocardial infarction functions
def filter_recurrent_cases_acute_myocardial_infarction(dataset):
    return filter_recurrent_cases(dataset, condition_name="acute_myocardial_infarction")

def doc_to_text_acute_myocardial_infarction(doc):
    return doc_to_text(doc, condition_name="acute_myocardial_infarction")

def doc_to_target_acute_myocardial_infarction(doc):
    return doc_to_target(doc, condition_name="acute_myocardial_infarction")

# Ischemic stroke functions
def filter_recurrent_cases_ischemic_stroke(dataset):
    return filter_recurrent_cases(dataset, condition_name="ischemic_stroke")

def doc_to_text_ischemic_stroke(doc):
    return doc_to_text(doc, condition_name="ischemic_stroke")

def doc_to_target_ischemic_stroke(doc):
    return doc_to_target(doc, condition_name="ischemic_stroke")

# Heart failure functions
def filter_recurrent_cases_heart_failure(dataset):
    return filter_recurrent_cases(dataset, condition_name="heart_failure")

def doc_to_text_heart_failure(doc):
    return doc_to_text(doc, condition_name="heart_failure")

def doc_to_target_heart_failure(doc):
    return doc_to_target(doc, condition_name="heart_failure")

# Dementia functions
def filter_recurrent_cases_dementia(dataset):
    return filter_recurrent_cases(dataset, condition_name="dementia")

def doc_to_text_dementia(doc):
    return doc_to_text(doc, condition_name="dementia")

def doc_to_target_dementia(doc):
    return doc_to_target(doc, condition_name="dementia")

# Intestinal cancer functions
def filter_recurrent_cases_intestinal_cancer(dataset):
    return filter_recurrent_cases(dataset, condition_name="intestinal_cancer")

def doc_to_text_intestinal_cancer(doc):
    return doc_to_text(doc, condition_name="intestinal_cancer")

def doc_to_target_intestinal_cancer(doc):
    return doc_to_target(doc, condition_name="intestinal_cancer")

# Gastric cancer functions
def filter_recurrent_cases_gastric_cancer(dataset):
    return filter_recurrent_cases(dataset, condition_name="gastric_cancer")

def doc_to_text_gastric_cancer(doc):
    return doc_to_text(doc, condition_name="gastric_cancer")

def doc_to_target_gastric_cancer(doc):
    return doc_to_target(doc, condition_name="gastric_cancer")

# Lung cancer functions
def filter_recurrent_cases_lung_cancer(dataset):
    return filter_recurrent_cases(dataset, condition_name="lung_cancer")

def doc_to_text_lung_cancer(doc):
    return doc_to_text(doc, condition_name="lung_cancer")

def doc_to_target_lung_cancer(doc):
    return doc_to_target(doc, condition_name="lung_cancer")

# Liver cancer functions
def filter_recurrent_cases_liver_cancer(dataset):
    return filter_recurrent_cases(dataset, condition_name="liver_cancer")

def doc_to_text_liver_cancer(doc):
    return doc_to_text(doc, condition_name="liver_cancer")

def doc_to_target_liver_cancer(doc):
    return doc_to_target(doc, condition_name="liver_cancer")

