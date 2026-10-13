import sys
from pathlib import Path

# Add project root directory to sys.path so 'src' can be imported anywhere
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.api import app
