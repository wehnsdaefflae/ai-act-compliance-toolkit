"""Allow running the toolkit as a module: python -m aiact_toolkit"""

import sys
from .cli import main

sys.exit(main())
