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
      model_train_data.csv
  processed/
    str/
    snp/

src/ancestry/
  config.py             # Default path cho task STR + POP hien tai
  data/                 # Loader/validation STR va SNP
  features/             # STRFeatureTransformer va SNPFeatureTransformer
  models.py             # Candidate models
  train.py              # Training pipeline theo config
  predict.py            # Prediction pipeline theo config

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

## Cach chay pipeline

Code dung mot pipeline chung cho cac task. Khac nhau giua STR/SNP va POP/SUBPOP nam trong file config:

```text
configs/str_pop.json
configs/str_subpop.json
configs/snp_pop.json
configs/snp_subpop.json
```

Train theo config:

```powershell
python scripts/train.py --config configs/str_pop.json
python scripts/train.py --config configs/snp_pop.json
```

Predict theo config:

```powershell
python scripts/predict.py --config configs/str_pop.json
python scripts/predict.py --config configs/snp_pop.json
```

Neu khong truyen `--config`, mac dinh van la task STR -> POP:

```powershell
python scripts/run_pipeline.py
```

Output duoc tach theo `output_dir` va `report_dir` trong tung config. Vi du SNP -> POP ghi vao:

```text
models/snp_pop/ancestry_snp_model.joblib
reports/snp_pop/cv_results.csv
reports/snp_pop/metrics.json
reports/snp_pop/holdout_predictions.csv
reports/snp_pop/predictions.csv
reports/snp_pop/run_metadata.json
reports/snp_pop/figures/confusion_matrix.png
```

Trong file SNP hien tai, loader se chuan hoa ten cot:

```text
sample_id  -> SAMPLE
super_pop  -> POP
pop        -> SUBPOP
rs...      -> SNP features
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

- Them feature selection cho SNP khi so marker lon hon.
- Them bao cao so sanh giua STR + POP va SNP + POP.
