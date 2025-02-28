import pandas as pd
import re
from embeddings import generate_text_embeddings

def extract_triplets_from_dataframe(data):
    """
    Extracts (entity, relation, attribute) triplets from the Radiopaedia dataset.

    Parameters:
        data (pd.DataFrame): A DataFrame containing the radiology dataset.

    Returns:
        list: A list of extracted triplets (Entity1, Relation, Entity2).
    """
    triplets = []

    for _, row in data.iterrows():
        disease = row['Disease']
        presentation = row.get('presentation', None)
        description = row.get('description', None)
        conclusion = row.get('conclusion', None)

        # Extract triplets for structured knowledge
        if pd.notna(presentation):
            triplets.append((disease, 'has symptom', presentation))

        if pd.notna(description):
            triplets.append((disease, 'described by', description))

        if pd.notna(conclusion):
            triplets.append((disease, 'concludes with', conclusion))

    return triplets

def clean_text(text):
    """Cleans input text by removing extra spaces and newlines."""
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def generate_triplet_embeddings(triplets):
    """
    Generates embeddings for extracted triplets.

    Parameters:
        triplets (list): List of extracted triplets.

    Returns:
        list: List of triplet embeddings.
    """
    triplet_embeddings = []
    for triplet in triplets:
        triplet_text = f"{triplet[0]} {triplet[1]} {triplet[2]}"
        triplet_embedding = generate_text_embeddings(triplet_text)
        triplet_embeddings.append(triplet_embedding)
    
    return triplet_embeddings

def save_triplets(triplets, file_path="triplets.csv"):
    """Saves extracted triplets into a CSV file."""
    triplets_df = pd.DataFrame(triplets, columns=['Entity1', 'Relation', 'Entity2'])
    triplets_df.to_csv(file_path, index=False)
    print(f"Triplets saved to {file_path}")

if __name__ == "__main__":
    # Load the Radiopaedia dataset
    file_path = 'Radiopaedia.xlsx'
    radiopaedia_data = pd.read_excel(file_path)

    # Extract triplets
    extracted_triplets = extract_triplets_from_dataframe(radiopaedia_data)
    print(f"Extracted {len(extracted_triplets)} triplets.")

    # Save triplets to a CSV file
    save_triplets(extracted_triplets)

    # Generate embeddings for triplets
    triplet_embeddings = generate_triplet_embeddings(extracted_triplets)
    print("Generated embeddings for triplets.")
    
    # Save triplet embeddings
    triplet_embeddings_np = {idx: emb for idx, emb in enumerate(triplet_embeddings)}
    np.save('triplet_embeddings.npy', triplet_embeddings_np)
    print("Triplet embeddings saved.")
