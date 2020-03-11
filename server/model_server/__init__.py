import os
import sys
from pathlib import Path

current_dir = Path(__file__).absolute().parent

# Add lib to PYTHONPATH
root_dir = current_dir.parents[2]
third_dir = root_dir / 'third'

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

for path in [root_dir, third_dir]:
    if path not in sys.path:
        sys.path.insert(0, str(path))
