import json
import pandas as pd

def clean_response(response_text):
    """Removes markdown artifacts from AI-generated JSON."""
    return response_text.strip("```json").strip("```")

def process_response(response_text):
    """Parses cleaned response into a structured DataFrame."""
    
    print("Raw Response Text:", response_text)  # 🚀 Debugging step

    if not response_text.strip():  # Check if response_text is empty
        raise ValueError("Error: response_text is empty!")

    cleaned_text = clean_response(response_text)
    
    try:
        response_data = json.loads(cleaned_text)  # Converts to dictionary
    except json.JSONDecodeError:
        raise ValueError("Error: response_text is not valid JSON!")

    # Convert dictionary to DataFrame
    df = pd.DataFrame([response_data])

    answer_row = [df[col][0]['answer'] for col in df.columns]
    source_row = [df[col][0]['sources'] for col in df.columns]
    reasoning_row = [df[col][0]['reasoning'] for col in df.columns]

    structured_response_df = pd.DataFrame(
        [answer_row, source_row, reasoning_row], columns=df.columns, index=['answer', 'source', 'reasoning']
    )

    return structured_response_df
