def doc_to_text_longstay(doc) -> str:
    """
    Generate the input prompt for operational task: inpatient longstay prediction.
    
    Args:
        doc: A dictionary containing the patient data with keys:
            - longstay_7d: inpatient visit longstay outcome (integer 0 or 1)
            - visit_cumulated: Full visit history text (prior visits only, before current admission)
            - visit_start_datetime: Current inpatient visit start datetime
    
    Returns:
        str: Formatted prompt text for the model
    """
    # Extract relevant information from the document
    visit_cumulated = doc["visit_cumulated"]
    current_visit_start = doc["visit_start_datetime"]
    
    # Create the prompt with task description before visit history
    prompt = f"""This is a task about predicting if the patient's current inpatient visit will be a long stay visit (staying more than 7 days) based on their prior EHR visit data: 

{visit_cumulated}

Above are the patient's prior EHR visit records (before the current admission) in chronological order. 

CURRENT ADMISSION INFORMATION:
- Current visit start datetime: {current_visit_start}
- The patient was admitted to the hospital as an inpatient visit.

You are a medical AI assistant analyzing patient data to predict if the patient's current inpatient visit will be a long stay visit (staying more than 7 days) given the patient's prior EHR visit records and current inpatient admission information provided above.

INSTRUCTIONS:
- Predict if the patient's current inpatient visit will be a long stay visit (staying more than 7 days)
- Do not provide any explanation, reasoning, or additional text
- You must respond with exactly one word only: Yes or No

Answer:"""
    
    return prompt


def doc_to_target_longstay(doc) -> str:
    """
    Extract the target label for the longstay prediction task.
    
    Args:
        doc: A dictionary containing the patient data with key:
            - longstay_7d: The actual inpatient visit longstay outcome (integer 0 or 1)
    
    Returns:
        str: "No" for 0, "Yes" for 1
    """
    # Convert integer (0 or 1) to string ("No"/"Yes") for generate_until tasks
    return "Yes" if int(doc["longstay_7d"]) == 1 else "No" 


def doc_to_text_readmission(doc) -> str:
    """
    Generate the input prompt for operational task: inpatient readmission prediction.
    
    Args:
        doc: A dictionary containing the patient data with keys:
            - readmission_30d: inpatient readmission outcome (integer 0 or 1)
            - visit_cumulated: Full visit history text (including index visit)
            - visit_start_datetime: Current inpatient visit start datetime
            - visit_end_datetime: Current inpatient visit end datetime (discharge time)
    
    Returns:
        str: Formatted prompt text for the model
    """
    # Extract relevant information from the document
    visit_cumulated = doc["visit_cumulated"]
    current_visit_end = doc["visit_end_datetime"]
    
    # Create the prompt with task description before visit history
    prompt = f"""This is a task about predicting if the patient will be readmitted to the hospital within 30 days after discharge from their current inpatient visit based on their EHR visit data:

{visit_cumulated}

Above are the patient's EHR visit records (including the current inpatient visit) in chronological order.



You are a medical AI assistant analyzing patient data to predict if the patient will be readmitted to the hospital within 30 days after discharge from their current inpatient visit given the patient's EHR visit records and current visit information provided above.

INSTRUCTIONS:
- Predict if the patient will be readmitted to the hospital within 30 days after the current inpatient visit end datetime ({current_visit_end})
- Do not provide any explanation, reasoning, or additional text
- You must respond with exactly one word only: Yes or No

Answer:"""
    
    return prompt


def doc_to_target_readmission(doc) -> str:
    """
    Extract the target label for the readmission prediction task.
    
    Args:
        doc: A dictionary containing the patient data with key:
            - readmission_30d: The actual inpatient readmission outcome (integer 0 or 1)
    
    Returns:
        str: "No" for 0, "Yes" for 1
    """
    # Convert integer (0 or 1) to string ("No"/"Yes") for generate_until tasks
    return "Yes" if int(doc["readmission_30d"]) == 1 else "No"




def doc_to_text_mortality(doc) -> str:
    """
    Generate the input prompt for 30-day mortality prediction task.
    
    Args:
        doc: A dictionary containing the patient data with keys:
            - visit_cumulated: Full visit history text
            - mortality_30d: Target outcome (0 = not deceased, 1 = deceased)
    
    Returns:
        str: Formatted prompt text for the model
    """
    # Extract relevant information from the document
    visit_history = doc["visit_cumulated"]
    
    # Create the prompt with task description before visit history
    prompt = f"""This is a task about predicting if the patient will die within the next 30 days after the start of the current inpatient hospitalization based on their EHR visit data: 

{visit_history}

Above are the patient's recent EHR visit records (including the current inpatient visit) in chronological order.

You are a medical AI assistant analyzing patient data to predict 30-day mortality given the patient's recent and current EHR visit records provided above.

INSTRUCTIONS:
- The patient is currently admitted as an inpatient (the current visit is an inpatient hospitalization)
- Predict if the patient will die within the next 30 days after this inpatient visit's start datetime
- Do not provide any explanation, reasoning, or additional text
- You must respond with exactly one word only: Yes or No

Answer:"""
    
    return prompt


def doc_to_target_mortality(doc) -> str:
    """
    Extract the target label for the mortality prediction task.
    
    Args:
        doc: A dictionary containing the patient data with key:
            - mortality_30d: The actual 30-day mortality outcome (0 = not deceased, 1 = deceased)
    
    Returns:
        str: "No" for 0, "Yes" for 1
    """
    # Convert integer (0 or 1) to string ("No"/"Yes") for generate_until tasks
    return "Yes" if int(doc["mortality_30d"]) == 1 else "No" 