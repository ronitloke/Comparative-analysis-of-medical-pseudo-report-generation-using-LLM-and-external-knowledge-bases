from google.cloud import storage

def initialize_storage_client():
    print("Initializing Google Cloud Storage client...")
    return storage.Client()

def list_and_sort_files(bucket_name, folder_name, file_extension, limit=None):
    """List and sort files from a Google Cloud Storage bucket."""
    client = initialize_storage_client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=folder_name))
    filtered_blobs = [blob for blob in blobs if blob.name.endswith(file_extension)]
    filtered_blobs.sort(key=lambda x: x.name)
    
    if limit:
        filtered_blobs = filtered_blobs[:limit]
    
    print(f"Found {len(filtered_blobs)} files in bucket: {bucket_name}")
    return filtered_blobs
