# sanity_mask_check.py
import pyarrow.ipc as pa_ipc
import numpy as np
from collections import Counter
from pathlib import Path
from omegaconf import OmegaConf

Config = OmegaConf.load("Config.yml")

ARROW_DIR = Config.dataset_path
ARROW_FILE = Config.teacher_answers_filename

path = Path(ARROW_DIR) / ARROW_FILE
with path.open("rb") as f:
    tbl = pa_ipc.open_file(f).read_all()

roles = tbl["student_role_ids"].to_pylist()
masks = tbl["student_loss_mask"].to_pylist()

# 1) infer assistant role id (the role id that appears the most where mask==1)
role_counts = Counter()
for r, m in zip(roles, masks):
    for rr, mm in zip(r, m):
        if mm == 1:
            role_counts[rr] += 1
assistant_role = role_counts.most_common(1)[0][0]
print(f"Inferred assistant role id: {assistant_role} (distribution: {role_counts})")

# 2) assert all mask==1 tokens are in assistant role
violations = 0
checked = 0
for r, m in zip(roles, masks):
    r = np.asarray(r)
    m = np.asarray(m)
    checked += int(m.sum())
    violations += int(((m == 1) & (r != assistant_role)).sum())

print(f"Checked masked tokens: {checked}, violations: {violations}")
if violations == 0:
    print("✅ All masked tokens are within the assistant role only.")
else:
    print("❌ Mask leakage detected (masked tokens outside assistant role).")

# 3) print a few boundaries for eyeballing
printed = 0
for idx, (r, m) in enumerate(zip(roles, masks)):
    m = np.asarray(m)
    if (m == 1).any():
        start = int(np.argmax(m == 1))
        end = len(m) - int(np.argmax(m[::-1] == 1)) - 1
        print(f"example {idx}: mask starts at {start}, ends at {end}")
        printed += 1
        if printed >= 3:
            break

