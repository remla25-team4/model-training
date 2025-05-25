import nltk

resources_to_download = ['wordnet', 'stopwords', 'omw-1.4'] # 'omw-1.4' is often needed with wordnet

for resource in resources_to_download:
    print(f"Attempting to download NLTK '{resource}'...")
    try:
        nltk.download(resource, quiet=True)
        print(f"'{resource}' downloaded successfully or already present.")
    except Exception as e:
        print(f"Error downloading '{resource}': {e}")
        print(f"Please try 'python -m nltk.downloader {resource}' manually in your terminal.")

print("\nNLTK resource download process complete.")