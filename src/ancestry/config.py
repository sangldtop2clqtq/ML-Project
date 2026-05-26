from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "ST1_HoanChinh.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"

TARGET_COLUMN = "POP"
ID_COLUMNS = ("SAMPLE", "POP", "SUBPOP")

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5

