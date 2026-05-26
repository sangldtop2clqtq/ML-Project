# Predicting Biogeographical Ancestry From STR DNA Data

Pipeline Machine Learning cho bài toán dự đoán nguồn gốc địa lý sinh học
(`POP`: AFR, AMR, EAS, EUR, SAS) từ dữ liệu STR genotype.

## Cấu trúc dự án

```text
data/
  raw/                  # File Excel gốc
  interim/
    ST1_HoanChinh.csv   # File ST1 đã xử lý, dùng để huấn luyện
  processed/            # Dữ liệu sinh ra từ pipeline nếu cần
src/ancestry/           # Mã nguồn pipeline
scripts/run_pipeline.py # Entry point huấn luyện
models/                 # Model đã train
reports/                # Metrics, prediction, hình confusion matrix
```

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chạy toàn bộ pipeline

```powershell
python scripts/run_pipeline.py
```

Pipeline sẽ:

1. đọc `data/interim/ST1_HoanChinh.csv`;
2. tự phát hiện các cặp allele `A1/A2`, `A1.1/A2.1`, ...;
3. chuẩn hóa genotype theo dạng không phân biệt thứ tự allele;
4. tạo feature cho mỗi locus: allele thấp, allele cao, tổng allele, chênh lệch allele, trạng thái dị hợp tử;
5. chia train/test có stratify theo `POP`;
6. so sánh baseline models bằng cross-validation;
7. train model tốt nhất;
8. lưu model, metrics, holdout predictions và confusion matrix.

## Output chính

```text
models/ancestry_str_model.joblib
reports/cv_results.csv
reports/metrics.json
reports/holdout_predictions.csv
reports/run_metadata.json
reports/figures/confusion_matrix.png
```

## Dự đoán với model đã train

```powershell
python scripts/predict.py --data-path data/interim/ST1_HoanChinh.csv
```

## Ghi chú khoa học

- `POP` là nhãn ancestry cấp nhóm lớn; `SUBPOP` có thể dùng ở giai đoạn sau để
  đánh giá chi tiết theo quần thể con.
- STR allele là dữ liệu genotype không có hướng, nên pipeline sắp xếp mỗi cặp
  allele thành `low/high` để tránh coi `10/11` và `11/10` là hai kiểu khác nhau.
- Các model hiện tại là baseline có thể giải thích và so sánh nhanh. Sau khi có
  kết quả đầu tiên, nên kiểm tra confusion matrix để xem các nhóm nào dễ bị nhầm,
  rồi mới tối ưu feature/model.
