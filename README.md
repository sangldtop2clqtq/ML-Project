# Predicting Biogeographical Ancestry From Genotype Data

Pipeline Machine Learning cho bai toan du doan nguon goc dia ly sinh hoc tu du lieu genotype.
Project duoc cau truc theo 4 task chinh:

```text
STR + POP
STR + SUBPOP
SNP + POP
SNP + SUBPOP
```

## Cau truc du an

```text
configs/
  str_pop.json          # STR genotype -> POP
  str_subpop.json       # STR genotype -> SUBPOP
  snp_pop.json          # SNP genotype -> POP
  snp_subpop.json       # SNP genotype -> SUBPOP

data/
  raw/
    str/                # File STR goc: ST1, ST2, ST3, ST4
    snp/                # File SNP goc se bo sung sau
  interim/
    str/
      str_genotypes_cleaned.csv
    snp/
      snp_genotypes_cleaned.csv
  processed/
    str/
    snp/

src/ancestry/
  data/                 # Vi tri danh cho loader/validation theo tung kieu genotype
  features/             # Vi tri danh cho feature transformer STR/SNP
  config.py             # Default path cho task STR + POP hien tai
  data.py               # Loader/validation STR hien tai
  features.py           # STRFeatureTransformer hien tai
  models.py             # Candidate models
  train.py              # Training pipeline hien tai
  predict.py            # Prediction pipeline hien tai

models/
  str_pop/
  str_subpop/
  snp_pop/
  snp_subpop/

reports/
  str_pop/
  str_subpop/
  snp_pop/
  snp_subpop/
```

## Cai dat

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Task hien tai dang chay

Code hien tai van dang ho tro truc tiep task:

```text
STR genotype -> POP
```

Du lieu mac dinh:

```text
data/interim/str/str_genotypes_cleaned.csv
```

Output mac dinh:

```text
models/str_pop/ancestry_str_model.joblib
reports/str_pop/cv_results.csv
reports/str_pop/metrics.json
reports/str_pop/holdout_predictions.csv
reports/str_pop/run_metadata.json
reports/str_pop/figures/confusion_matrix.png
```

Chay pipeline hien tai:

```powershell
python scripts/run_pipeline.py
```

Du doan voi model da train:

```powershell
python scripts/predict.py --data-path data/interim/str/str_genotypes_cleaned.csv
```

## Y nghia 4 config

```text
configs/str_pop.json
```

Dung STR genotype de du doan nhan ancestry cap nhom lon `POP`.

```text
configs/str_subpop.json
```

Dung STR genotype de du doan nhan quan the con `SUBPOP`.

```text
configs/snp_pop.json
```

Dung SNP genotype de du doan nhan ancestry cap nhom lon `POP`.

```text
configs/snp_subpop.json
```

Dung SNP genotype de du doan nhan quan the con `SUBPOP`.

## Huong phat trien tiep

- Tach logic hien tai trong `data.py` thanh `src/ancestry/data/str_loader.py`.
- Tach `STRFeatureTransformer` trong `features.py` thanh `src/ancestry/features/str_features.py`.
- Them `src/ancestry/data/snp_loader.py` va `src/ancestry/features/snp_features.py`.
- Them `registry.py` de chon loader/feature transformer theo `genotype_type`.
- Cap nhat `train.py` va `predict.py` de doc config bang tham so `--config`.
