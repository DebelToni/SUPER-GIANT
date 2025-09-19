from transformers import AutoTokenizer
from omegaconf import OmegaConf

Config = OmegaConf.load("Teacher_config.yml")

# Load and save the tokenizer
tok = AutoTokenizer.from_pretrained(Config.model_tokenizer, use_fast=True)
tok.save_pretrained(Config.student_tokenizer_dir)
