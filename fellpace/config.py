# define a constant which is equal to the path of the project file

import os
from pathlib import Path

# get the path of the project file
PROJECT_PATH = Path(os.path.abspath(os.path.join(os.path.dirname(__file__))))
DB_DIR = PROJECT_PATH.parent / "DB"
DB_PATH = DB_DIR / "fellpace.db"
ENTRIES_PATH = PROJECT_PATH.parent / "entries"

MODELS_PATH = PROJECT_PATH.parent / "models"
COEFFS_FILE_PATH = MODELS_PATH / "coeffs.json"
COVAR_FILE_PATH = MODELS_PATH / "covars.json"

# Race results
EXCLUDE_LIST = ['Exterminator']