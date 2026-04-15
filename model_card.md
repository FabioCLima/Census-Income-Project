# Model Card: Census Income Classification

## Model Details

- Model type: `RandomForestClassifier` inside a full sklearn `Pipeline`.
- Target: binary income class (`>50K` = 1, `<=50K` = 0).
- Positive class for metrics: `>50K`.
- Training script: `train_model.py` (entrypoint to `census.train_model.run_training_pipeline`).
- Stored artifacts:
  - `model/census_pipeline.pkl` (preprocessor + model)
  - `model/model.pkl` (estimator only)
  - `model/encoder.pkl` (fitted categorical `OrdinalEncoder`)

## Intended Use

- Primary use: educational MLOps workflow and API inference exercises.
- Supported input: one-row payload compatible with `PredictRequest`.
- Output: predicted class label (`>50K` or `<=50K`).
- Not intended for high-stakes decisions (credit, hiring, legal, healthcare).

## Training Data

- Dataset: Adult Census Income (cleaned project version).
- Records used: `32,561` rows after cleaning.
- Features used by the pipeline: `13` (6 numeric + 7 categorical).
- Split strategy: stratified train/test split (`80/20`) to preserve class ratio.
- Class distribution (global): approximately `24.1%` positive (`>50K`).

## Metrics

### Cross-validation (training set only, 5-fold StratifiedKFold)

Source: `model/cv_results.csv`.

- Decision Tree baseline (mean):
  - Precision: `0.5230`
  - Recall: `0.8573`
  - F1: `0.6493`
- Random Forest candidate (mean):
  - Precision: `0.7505`
  - Recall: `0.6067`
  - F1: `0.6710`

Model selected: `RandomForestClassifier` (higher mean F1).

### Slice metrics (test split)

Source: `model/slice_output.txt`.

- `sex=Female`: precision `0.8079`, recall `0.5837`, F1 `0.6777`
- `sex=Male`: precision `0.7524`, recall `0.6387`, F1 `0.6909`

## Bias Assessment (Aequitas)

Source: `notebooks/bias_census_study.ipynb` (executed on 2026-04-14).

Method:

- Package: `aequitas`
- Attributes evaluated: `sex`, `race`
- Reference groups: `Male` (sex), `White` (race)
- Screening rule: disparity outside `[0.8, 1.25]` (80% rule)
- Stability filter used in interpretation: `group_size >= 100`

Visual summary:

### Stable sex groups (reference = `Male`)

| Group | Size | FPR disp. | FNR disp. | FOR disp. | FDR disp. | 80% rule |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Female | 2158 | 0.1938 | 1.1523 | 0.3481 | 0.7760 | flagged |
| Male | 4355 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | reference |

### Stable race groups (reference = `White`)

| Group | Size | FPR disp. | FNR disp. | FOR disp. | FDR disp. | 80% rule |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Asian-Pac-Islander | 200 | 1.1567 | 1.0004 | 1.0352 | 1.0925 | within range |
| Black | 662 | 0.2758 | 1.1776 | 0.5649 | 0.7042 | flagged |
| White | 5533 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | reference |

Interpretation:

- `Female` is outside the acceptable disparity band in multiple metrics, especially `fpr_disparity`, `for_disparity`, and `fdr_disparity`.
- `Black` is the race group most clearly flagged in the stable-size subset, with disparity values below `0.8` in multiple metrics.
- `Asian-Pac-Islander` remains within the screening range in this run.
- Some race groups with very small support do not appear in the stable summary and should be treated as statistically unstable rather than ignored.

Conclusion:

- The current model presents **measurable group disparity (bias risk)**, especially across `sex` and some `race` slices.

## Ethical Considerations

- The Adult dataset contains sensitive demographic attributes and historical bias.
- Similar global metrics do not guarantee equitable behavior across all groups.
- Slice monitoring is required before any deployment-like usage.
- The model can reinforce patterns present in historical income data.

## Limitations

- Dataset is historical and not representative of current labor markets.
- Performance can vary for underrepresented slices.
- This project prioritizes reproducible workflow and validation, not production fairness guarantees.
- Calibration and threshold tuning were not optimized for domain-specific cost trade-offs.

## Maintenance and Monitoring

- Re-train when data schema or upstream preprocessing changes.
- Keep `model/cv_results.csv` and `model/slice_output.txt` updated after each retraining cycle.
- Track drift in categorical distributions and slice-level degradation over time.
