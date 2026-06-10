# ============================================================
# PASTE THIS INTO A NEW CELL AT THE END OF YOUR v5 COLAB
# Run it after cell 7 (train all 4 appliances)
# It saves the 4 .pt files and lets you download them
# ============================================================

import shutil, os
from google.colab import files

EXPORT_DIR = "/content/nilm_export"
os.makedirs(EXPORT_DIR, exist_ok=True)

for app in cfg.APPLIANCES:
    src = os.path.join(cfg.CHECKPOINT_DIR, f"v5_{app}.pt")
    dst = os.path.join(EXPORT_DIR, f"nilm_{app}.pt")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        size_kb = os.path.getsize(dst) / 1024
        print(f"✅ {dst}  ({size_kb:.0f} KB)")
    else:
        print(f"⚠ {src} not found — did training finish?")

# Download all 4 files to your computer
for app in cfg.APPLIANCES:
    path = os.path.join(EXPORT_DIR, f"nilm_{app}.pt")
    if os.path.exists(path):
        files.download(path)
        print(f"⬇ Downloading nilm_{app}.pt")
