from pathlib import Path

from transformers import AutoTokenizer
from omegaconf import OmegaConf

DATA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DATA_DIR.parent

cfg = OmegaConf.merge(
    OmegaConf.load(PROJECT_ROOT / "Global_Config.yml"),
    OmegaConf.load(DATA_DIR / "Config.yml"),
)

teacher_cfg = cfg.teacher if "teacher" in cfg else cfg
tokenizer_id = getattr(teacher_cfg, "model_tokenizer", cfg.tokenizer.name)
output_dir = getattr(teacher_cfg, "student_tokenizer_dir", "student_tokenizer")

# Load and save the tokenizer
tok = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
tok.save_pretrained(output_dir)
