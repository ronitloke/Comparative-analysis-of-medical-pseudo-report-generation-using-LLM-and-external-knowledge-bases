import re

def clean_text(text):
    """Cleans text by removing extra spaces and newlines."""
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def save_to_file(data, file_path):
    """Saves data to a file."""
    with open(file_path, 'w') as file:
        for item in data:
            file.write(f"{item}\n")
