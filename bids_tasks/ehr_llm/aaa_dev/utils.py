
def doc_to_text_simple(doc) -> str:
    """
    Simple prompt generation for testing.
    
    Args:
        doc: A dictionary containing the prompt data
    
    Returns:
        str: The prompt text
    """
    return doc.get("prompt", "")


def doc_to_target_simple(doc) -> str:
    """
    Simple target extraction for testing.
    
    Args:
        doc: A dictionary containing the answer data
    
    Returns:
        str: The target answer
    """
    return doc.get("answer", "") 

