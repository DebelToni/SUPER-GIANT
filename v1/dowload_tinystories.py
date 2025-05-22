from datasets import load_dataset
data = load_dataset("roneneldan/TinyStories")
print(data.keys())  # should show 'train' and 'validation' splits

