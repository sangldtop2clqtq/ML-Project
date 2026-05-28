from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- EXTENSION POINT: CONFIGURATION PATHS & TARGETS ---
# Currently set up for the default task: STR -> POP.
# To extend to other tasks (STR -> SUBPOP, SNP -> POP, SNP -> SUBPOP):
# 1. STR -> SUBPOP: 
#    - Change TARGET_COLUMN to "SUBPOP"
#    - Update DEFAULT_OUTPUT_DIR to PROJECT_ROOT / "models" / "str_subpop"
#    - Update DEFAULT_REPORT_DIR to PROJECT_ROOT / "reports" / "str_subpop"
# 2. SNP -> POP:
#    - Update DEFAULT_DATA_PATH to PROJECT_ROOT / "data" / "interim" / "snp" / "snp_genotypes_cleaned.csv"
#    - Update DEFAULT_OUTPUT_DIR to PROJECT_ROOT / "models" / "snp_pop"
#    - Update DEFAULT_REPORT_DIR to PROJECT_ROOT / "reports" / "snp_pop"
#    - Change TARGET_COLUMN to "POP"
# 3. SNP -> SUBPOP:
#    - Update DEFAULT_DATA_PATH to PROJECT_ROOT / "data" / "interim" / "snp" / "snp_genotypes_cleaned.csv"
#    - Update DEFAULT_OUTPUT_DIR to PROJECT_ROOT / "models" / "snp_subpop"
#    - Update DEFAULT_REPORT_DIR to PROJECT_ROOT / "reports" / "snp_subpop"
#    - Change TARGET_COLUMN to "SUBPOP"
# Tip: Alternatively, you can read these values dynamically from JSON files in the configs/ folder.

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "str" / "str_genotypes_cleaned.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "str_pop"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "str_pop"

TARGET_COLUMN = "POP"
ID_COLUMNS = ("SAMPLE", "POP", "SUBPOP")

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5

