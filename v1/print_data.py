from datasets import load_dataset, config
data = load_dataset("roneneldan/TinyStories",
                    cache_dir="/Users/antonhristov/.cache/huggingface/datasets/roneneldan___tiny_stories/default/0.0.0/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
)

print(data)
