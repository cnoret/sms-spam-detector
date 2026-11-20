import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# app.py and train.py use paths relative to the repository root.
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
