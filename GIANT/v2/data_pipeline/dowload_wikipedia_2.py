import kagglehub

# Download latest version
path = kagglehub.dataset_download("ffatty/plaintext-wikipedia-full-english")

print("Path to dataset files:", path)
