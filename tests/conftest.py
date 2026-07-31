import sys
from pathlib import Path

# Make `import unirl` work without installing the package (the CPU-only tests
# import ray/torch-free modules only).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
