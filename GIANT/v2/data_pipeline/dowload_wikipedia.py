import kagglehub

# Download latest version
path = kagglehub.dataset_download("ltcmdrdata/plain-text-wikipedia-202011")

print("Path to dataset files:", path)
