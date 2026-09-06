# 🛒 Customer Purchase Prediction 

Proyek machine learning untuk memprediksi apakah pengunjung sebuah situs e-commerce akan melakukan pembelian (`Revenue`) berdasarkan perilaku mereka selama browsing, menggunakan **Random Forest** dan **XGBoost** dengan penanganan class imbalance via SMOTE.

## 🎯 Business Problem

Tidak semua visitor pada situs e-commerce melakukan pembelian, sehingga sulit untuk mengidentifikasi visitor yang berpotensi membeli serta memahami faktor yang memengaruhi keputusan pembelian mereka.

**Tujuan:** Memprediksi apakah seorang pengunjung website berpotensi melakukan pembelian berdasarkan perilaku mereka selama mengunjungi website, sekaligus memahami faktor-faktor yang paling berpengaruh.

**Target variable:** `Revenue`
- `True` = melakukan pembelian
- `False` = tidak melakukan pembelian

## 📊 Dataset

Dataset: [Online Shoppers Intention](https://www.kaggle.com/datasets/henrysue/online-shoppers-intention) (Kaggle) 12.330 baris data sesi kunjungan e-commerce.

| Tipe | Kolom |
|---|---|
| Numerik | `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`, `OperatingSystems`, `Browser`, `Region`, `TrafficType` |
| Kategorikal | `Month`, `VisitorType` |
| Boolean | `Weekend`, `Revenue` (target) |

## 🔄 Alur 

1. **Data Understanding**: cek struktur, tipe data, missing value, dan duplikat.
2. **EDA (Raw Data)**: distribusi target, korelasi fitur numerik terhadap `Revenue`, conversion rate per `VisitorType`/`Weekend`/`Month`, deteksi outlier ekstrem pada fitur durasi menggunakan percentile check (bukan IQR, karena data skewed).
3. **Data Cleaning**: hapus duplikat, konversi kolom boolean ke integer.
4. **EDA (Clean Data)**: validasi ulang statistik dan korelasi setelah cleaning.
5. **Data Preprocessing**: train-test split (80:20, stratified), one-hot encoding untuk `Month` dan `VisitorType` (fit dari train, lalu disesuaikan ke test agar tidak bocor), Outlier Handling (capping pada persentil ke-99 untuk fitur durasi).
7. **Modeling**: Random Forest dan XGBoost, masing-masing dalam pipeline `SMOTE → classifier`, dituning dengan `RandomizedSearchCV` (scoring `f1`, `StratifiedKFold` 5-fold).
8. **Evaluasi**: accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, ROC curve, precision-recall curve.
9. **Feature Importance**: perbandingan fitur paling berpengaruh dari kedua model.

## 🏆 Hasil 

| Metrik | Random Forest | XGBoost |
|---|---|---|
| F1-Score | 0.6958 | 0.6875 |
| ROC-AUC | 0.9331 | 0.9334 |
| Accuracy | ~89–90% | ~89–90% |
| Precision (kelas pembeli) | ~0.64–0.65 | ~0.64–0.65 |
| Recall (kelas pembeli) | ~0.75 | ~0.75 |

**Insight:**
- Terdapat class imbalance signifikan: dari 12.205 data bersih, hanya 15,6% (1.908 visitor) yang melakukan pembelian.
- `PageValues` adalah fitur dengan korelasi positif terkuat terhadap `Revenue` (0,49) dan konsisten menjadi fitur paling penting di kedua model.
- `ExitRates` berkorelasi negatif (-0,20) terhadap pembelian.
- `New Visitor` memiliki conversion rate lebih tinggi (24,9%) dibanding `Returning Visitor` (13,9%), meski jumlahnya lebih sedikit.
- Bulan November mencatat conversion rate tertinggi (25,35%), Februari terendah (1,63%).

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
kaggle
```
