import sys
from pathlib import Path

# Put src/ on the path so `import nmc` works without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
