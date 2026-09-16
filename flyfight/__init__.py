"""Connectome-constrained research prototype; not a validated biological replica."""
__version__ = '0.1.0'
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent.parent/'results'/'.mplcache'))
