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
RESID_STD_FILE_PATH = MODELS_PATH / "resid_stds.json"
TIME_COEFFS_FILE_PATH = MODELS_PATH / "time_coeffs.json"
ROAD_TIME_COEFFS_FILE_PATH = MODELS_PATH / "road_time_coeffs.json"

# Race results
EXCLUDE_LIST = ['Exterminator']

# Race details
START_TIME = "19:30:00"