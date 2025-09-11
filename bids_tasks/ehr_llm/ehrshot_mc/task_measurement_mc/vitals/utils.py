def doc_to_text(doc) -> str:
    """
    Generate the input prompt for vital signs prediction task.
    
    Args:
        doc: A dictionary containing the patient data with keys:
            - visit_cumulated: Full visit history text
            - measurement_concept_name: Name of the vital sign measurement
            - current_value: Current vital sign value
            - threshold: Threshold value for normal/abnormal classification
            - unit: Unit of measurement
            - normal_direction: Direction for normal classification ("below", "above", or "in range")
    
    Returns:
        str: Formatted prompt text for the model
    """
    # Extract relevant information from the document
    visit_history = doc["visit_cumulated"]
    measurement_name = doc["measurement_concept_name"]
    current_value = doc["current_value"]
    threshold = doc["threshold"]
    unit = doc["unit"]
    normal_direction = doc["normal_direction"]
    
    # Handle different threshold types and create appropriate descriptions
    if normal_direction == "below":
        # For measurements where lower values are normal (e.g., blood pressure)
        threshold_explanation = f"For {measurement_name}, normal values are {threshold} {unit} or below, while abnormal values are above {threshold} {unit}."
    elif normal_direction == "above":
        # For measurements where higher values are normal (rare for vitals)
        threshold_explanation = f"For {measurement_name}, normal values are {threshold} {unit} or above, while abnormal values are below {threshold} {unit}."
    elif normal_direction == "in range":
        # For measurements where values within a range are normal (e.g., heart rate)
        # threshold should be a string like "[60, 100]" - we need to parse it
        threshold_str = threshold.strip("[]")
        min_val, max_val = threshold_str.split(", ")
        threshold_explanation = f"For {measurement_name}, normal values are between {min_val} and {max_val} {unit} (inclusive), while abnormal values are below {min_val} or above {max_val} {unit}."
    
    # Create the prompt with task description before visit history
    prompt = f"""This is a task about predicting if the patient's next {measurement_name} measurement will be normal or abnormal based on their recent EHR visit data: 

{visit_history}

Above are the patient's recent EHR visit records (including the current visit) in chronological order.

You are a medical AI assistant predicting patient {measurement_name} trends given the patient's recent and current EHR visit records provided above. The {measurement_name} value in the current visit is {current_value} {unit}.

INSTRUCTIONS:
- Predict if the next available {measurement_name} measurement in the following visits will be abnormal
- {threshold_explanation}
- Consider the patient's medical history, current vital sign value, and clinical trends when making your prediction
- Do not provide any explanation, reasoning, or additional text
- You must respond with exactly one word only: Yes or No

Answer:"""
    
    return prompt


def doc_to_target(doc) -> int:
    """
    Extract the target label for the vital signs prediction task.
    
    Args:
        doc: A dictionary containing the patient data with key:
            - next_label: The actual next measurement label (0 for "normal", 1 for "abnormal")
    
    Returns:
        int: 0 for "No", 1 for "Yes"
    """
    # The next_label is already an integer (0 or 1) in the new schema
    return doc["next_label"]
