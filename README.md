# Liver Cancer Classification — Microarray Gene Expression ML Project

This project builds machine learning models to classify liver tissue samples as either Hepatocellular Carcinoma (HCC) or normal liver tissue using gene expression data from Affymetrix microarray profiling. The dataset comes from a published GEO study (GSE14520) and contains 357 patient samples, each described by the expression levels of 22,277 gene probes. The goal is to train a model that can correctly distinguish cancerous from healthy liver tissue based solely on those expression patterns.

**Note:** This is an introductory project intended to help students understand the core concepts and workflow of applied machine learning — data loading, exploratory analysis, preprocessing, feature selection, model training, hyperparameter tuning, and evaluation. The focus is on understanding the process clearly, not achieving state-of-the-art performance.

The project is structured as four sequential Jupyter notebooks, each handling one stage of the pipeline. The notebooks are designed for students learning applied machine learning, and every step is explained in plain language alongside the code.

---

## Table of Contents

- [Background](#background)
- [The Dataset](#the-dataset)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Notebook 1 — EDA and Loading](#notebook-1--eda-and-loading)
- [Notebook 2 — Preprocessing and Feature Reduction](#notebook-2--preprocessing-and-feature-reduction)
- [Notebook 3 — Logistic Regression](#notebook-3--logistic-regression)
- [Notebook 4 — Gradient Boosting and Model Comparison](#notebook-4--gradient-boosting-and-model-comparison)
- [Key Design Decisions](#key-design-decisions)
- [Libraries Used](#libraries-used)
- [Results Summary](#results-summary)

---

## Background

### What is Hepatocellular Carcinoma?

Hepatocellular Carcinoma (HCC) is the most common form of primary liver cancer and one of the leading causes of cancer-related death worldwide. It typically develops in people who already have liver disease — most commonly cirrhosis caused by chronic hepatitis B or C infection, heavy alcohol use, or non-alcoholic fatty liver disease. Because HCC often develops silently and symptoms appear late, early and accurate detection is critical for improving patient outcomes.

Diagnosing HCC currently involves imaging scans, blood tests for a protein called alpha-fetoprotein (AFP), and tissue biopsies. All of these methods have limitations — biopsies are invasive, AFP is not always elevated in HCC, and imaging can miss small tumours. Gene expression profiling offers a complementary molecular approach: rather than looking at the shape or size of the tissue, it asks which genes are switched on or off inside it.

### What is gene expression profiling?

Every cell in the human body contains the same DNA, but different cells use different parts of it. A liver cell activates a very different set of genes compared to a blood cell or a skin cell. When a cell becomes cancerous, the pattern of gene activation changes — some genes that should be silent become active, and some that should be active are switched off. These changes are detectable through gene expression profiling.

The technology used in this dataset is an **Affymetrix microarray**. A microarray is a glass chip covered with thousands of short DNA sequences (probes), each designed to bind to the RNA produced by one specific gene. When a patient's tissue sample is applied to the chip, the amount of binding at each probe location indicates how much of that gene's RNA is present. The result is a numeric expression value for each of the 22,277 probes on the chip — a molecular fingerprint of the tissue.

### Why machine learning?

With over 22,000 numeric features per sample and only 357 samples, this is a classic high-dimensional, small-sample problem. Traditional statistical tests can identify individual genes that differ between HCC and normal tissue, but they consider each gene in isolation. Machine learning models can find combinations of genes — patterns across many features simultaneously — that predict cancer status more accurately than any single gene alone. The challenge is to do this without overfitting: with far more features than samples, a model that simply memorises the training data will perform poorly on new patients.

---

## The Dataset

**File:** `Liver_GSE14520_U133A.csv`  
**Source:** NCBI Gene Expression Omnibus — accession GSE14520, platform GPL571  
**Rows:** 357 (one row per patient tissue sample)  
**Columns:** 22,279 total — 2 metadata columns + 22,277 gene probe columns

### Metadata Columns

| Column | Type | Description |
|---|---|---|
| `samples` | Text | GEO sample identifier (e.g. `GSM362958.CEL.gz`) |
| `type` | Text | **The target variable** — either `HCC` or `normal` |

### Gene Probe Columns

The remaining 22,277 columns are named after Affymetrix probe identifiers (e.g. `1007_s_at`, `1053_at`, `AFFX-ThrX-5_at`). Each value is a continuous floating-point number representing the log-normalised expression level of that gene in that sample. Unlike raw RNA counts, these values have already been pre-processed and normalised by the GEO data submission — there are no zero-inflation or sparsity issues as seen in scRNA-seq data.

### Class Distribution

| Class | Samples | Percentage |
|---|---|---|
| HCC | 181 | 50.7% |
| normal | 176 | 49.3% |

The dataset is almost perfectly balanced. HCC and normal samples contribute nearly equal numbers of rows. This means there is no class imbalance problem — we do not need to use class weighting, oversampling, or any other imbalance correction technique.

### Data Quality

- **Missing values:** None. Every probe has a value for every sample.
- **Data type:** All 22,277 gene columns are continuous floats — no categorical features outside the target.
- **Expression range:** Values are log-scale, typically ranging from approximately 3 to 14. This is expected for pre-normalised Affymetrix data and confirms the data has been processed correctly.

### What GPL571 means

GPL571 is the identifier for the Affymetrix Human Genome U133A 2.0 Array. This array covers approximately 14,500 human genes using 22,277 probe sets. The probe identifier column names (e.g. `1007_s_at`) are the internal Affymetrix IDs for each probe location on the chip. Each can be mapped to a gene symbol (like `MYC` or `TP53`) using Affymetrix annotation files, though this project uses the probe IDs directly without remapping.

### About GSE14520

GSE14520 is a well-cited study in the liver cancer gene expression literature. The samples come from patients who underwent surgical resection at a major hepatology centre. Half the samples are tumour tissue taken directly from the HCC mass; the other half are non-tumour liver tissue taken from the same patients, sampled away from the tumour site. This paired design means both classes come from the same cohort of patients, which reduces confounding factors and makes the classification task a genuine test of whether gene expression patterns differ between the tumour and the surrounding tissue.

---

## Project Structure

```
Cumida-ML-Model/
│
├── Liver_GSE14520_U133A.csv       ← raw dataset (357 samples, 22,277 probes)
│
├── 01_eda_loading.ipynb           ← data loading, inspection, EDA
├── 02_preprocessing.ipynb         ← feature reduction, train/test split
├── 03_logistic_regression.ipynb   ← Logistic Regression classifier
└── 04_gradient_boosting.ipynb     ← Gradient Boosting + model comparison
```

The notebooks must be run **in order**. Each notebook saves its outputs to Google Drive, and the next notebook reads them as inputs.

```
01_eda_loading.ipynb
    └── saves: liver_clean.csv

02_preprocessing.ipynb
    └── reads: liver_clean.csv
    └── saves: X_train.csv
                X_test.csv
                y_train.csv
                y_test.csv

03_logistic_regression.ipynb
    └── reads: X_train.csv, X_test.csv, y_train.csv, y_test.csv

04_gradient_boosting.ipynb
    └── reads: X_train.csv, X_test.csv, y_train.csv, y_test.csv
```

---

## How to Run

### Requirements

- A Google account with Google Drive
- Google Colab (free, runs in your browser — no local installation needed)
- The dataset file `Liver_GSE14520_U133A.csv` uploaded to a folder on your Google Drive

### Setup Steps

1. Create a folder on your Google Drive, for example: `My Drive/Cumida-ML-Model/`
2. Upload `Liver_GSE14520_U133A.csv` into that folder
3. Open each notebook in Google Colab (File → Open notebook → Google Drive, or upload from GitHub)
4. In **each notebook**, find the configuration cell near the top and update `DATA_DIR` to match your folder path:

```python
DATA_DIR = Path('/content/drive/MyDrive/Cumida-ML-Model')
```

5. Run the notebooks in order: `01` → `02` → `03` → `04`

### Runtime Notes

- **Notebook 1** runs in under a minute. The main bottleneck is loading the 22,277-column CSV.
- **Notebook 2** is the slowest notebook. The RFE sweep trains many Random Forest models internally. On Google Colab's free tier, expect it to take 10–25 minutes depending on how many features remain after VarianceThreshold.
- **Notebook 3** (Logistic Regression + GridSearchCV) typically runs in 2–5 minutes.
- **Notebook 4** (Gradient Boosting + BayesSearchCV) typically runs in 5–15 minutes.

---

## Notebook 1 — EDA and Loading

**File:** `01_eda_loading.ipynb`  
**Input:** `Liver_GSE14520_U133A.csv`  
**Output:** `liver_clean.csv`

This notebook handles all data loading and exploratory analysis. No machine learning happens here — the goal is to understand the dataset thoroughly before preprocessing or modelling.

### What it does, step by step

**Step 1 — Load the raw dataset**  
Reads the CSV file, prints the shape, confirms the class labels and gene probe count.

**Step 2 — Initial inspection**  
Checks data types across all columns, confirms that all 22,277 gene probe columns are numeric, reports the total number of missing values, and prints descriptive statistics for a sample of probes.

**Step 3 — Class distribution**  
Prints the count and percentage of HCC vs normal samples. Produces two plots side by side: a bar chart with sample counts labelled above each bar, and a pie chart showing the class proportions.

**Step 4 — Missing value analysis**  
Computes the percentage of missing values per probe across all samples. If any probes have missing values, a bar chart of the top 50 affected probes is shown with reference lines at 50% and 90%. For this particular dataset, there are no missing values, so this section confirms data completeness.

**Step 5 — Top variable genes: expression distributions by class**  
Selects the top 20 gene probes ranked by variance across all samples (these are the probes that vary the most between samples and are therefore most likely to be informative for classification). For each of these 20 probes, an overlapping histogram is plotted showing the distribution of expression values separately for HCC and normal samples. Probes where the two distributions are clearly separated are strong candidate features for the model.

**Step 6 — Box plots by class**  
The same top 20 probes are shown as box plots grouped by class. Box plots complement the histograms by making the median, interquartile range, and outliers more visible.

**Step 7 — Correlation heatmap**  
A correlation heatmap is computed for the top 30 most variable probes. The lower triangle of the matrix is shown using a coolwarm colour scale. Strong positive correlations (red) indicate probes that tend to move together; strong negative correlations (blue) indicate probes that tend to move in opposite directions. Highly correlated probes carry redundant information and may be reduced during feature selection.

**Step 8 — Overall expression distribution**  
Two plots are produced side by side. The first is a histogram of a random sample of 500,000 individual expression values drawn from across the entire matrix — this confirms the expected log-scale distribution of pre-normalised Affymetrix data. The second shows the per-sample median expression value, plotted separately for HCC and normal samples, as a sanity check that neither class has systematically different overall expression levels.

**Step 9 — Save cleaned dataset**  
Drops the `samples` column (the GEO sample identifier is not a useful feature for modelling) and saves the result as `liver_clean.csv`. The shape is printed to confirm the save was successful.

---

## Notebook 2 — Preprocessing and Feature Reduction

**File:** `02_preprocessing.ipynb`  
**Input:** `liver_clean.csv`  
**Outputs:** `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`

This notebook prepares the data for machine learning. Its primary job is to reduce the 22,277 gene probes down to a small, informative subset, then save the final train and test sets.

### Data Leakage

A key principle throughout this notebook is **preventing data leakage**. Data leakage happens when information from the test set influences any part of the preprocessing or training process. For example, if you compute the mean and standard deviation of the full dataset (including the test set) and use those values to scale the features, the model has indirectly "seen" the test set before evaluation. This produces unrealistically optimistic performance numbers that will not hold up on genuinely new data.

To prevent this, every preprocessing step is **fitted on the training set only** and then **applied** to both the training and test sets using the parameters learned from the training data alone.

### What it does, step by step

**Step 1 — Separate features and target, then split**  
Creates `X` (the 22,277 gene probe columns) and `y` (the `type` column). A stratified 80/20 train/test split is applied using `random_state=42` to make it reproducible. Stratified means both the training and test sets preserve the ~50/50 HCC-to-normal ratio from the full dataset.

**Step 2 — Feature reduction pipeline**

The 22,277 gene probes are reduced in four sequential steps. The table below summarises the pipeline:

| Step | Technique | What it removes |
|---|---|---|
| 1 | Zero-variance filter | Probes with identical values across all training samples |
| 2 | High-null filter | Probes missing in more than 90% of training samples |
| 3 | Variance Threshold (VT) | Probes with low variance after scaling |
| 4 | RFE | Least informative probes, as ranked by a Random Forest |

**Step 1 — Zero-variance filter**  
Any probe that has the exact same numeric value in every single training sample carries no information — it cannot help distinguish HCC from normal tissue. These probes are identified by computing variance on the training set and dropping any probe where variance equals zero. This is applied to both train and test sets using the list of zero-variance probes identified from the training set only.

**Step 2 — High-null filter**  
Probes where more than 90% of training samples have a missing value are dropped. There are very few (or zero) such probes in this dataset, but the step is included for completeness and as a general-purpose safeguard. Any probes that survive but still have some missing values are imputed using the column median, calculated from the training set only.

**Step 3 — Variance Threshold**  
First, `StandardScaler` is applied to centre all probes to mean 0 and standard deviation 1. This is necessary so that probes with naturally large raw values do not dominate the variance calculation in the next step. Then `VarianceThreshold(threshold=0.01)` is applied: any probe whose variance across training samples falls below 0.01 (after scaling) is removed. Both the scaler and the threshold are fitted on the training data only. The `get_support()` method returns a boolean mask (True = keep) that is applied identically to both sets.

**Step 4 — RFE (Recursive Feature Elimination)**  
RFE works by fitting a model, scoring each feature by its importance, removing the least important features, and repeating until a target number of features `k` remains. To find the best `k`, a sweep is run over a set of candidate values (10, 20, 30, 50, 75, 100, 150, 200). For each candidate, RFE selects those features, a Random Forest is evaluated on them using 5-fold cross-validation, and the mean F1 score is recorded. The results are plotted as a line chart of CV F1 vs number of features, with the best `k` marked by a red dashed vertical line. The final RFE is then re-run using the best `k` and a more robust 100-tree Random Forest as the estimator.

**Why is a Random Forest used inside RFE?**  
RFE needs a model that produces a feature importance score for each feature after fitting. Random Forest provides `feature_importances_` natively — each probe is scored by how much it reduces prediction error across all trees. Logistic Regression and Gradient Boosting can also be used inside RFE, but Random Forest is a reliable and fast default for this ranking step. The genes selected here are then passed to both downstream models as the final feature set.

**Step 3 — Save outputs**  
The four files are saved: `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`. A summary table is printed showing the feature count at each step, how many probes were removed at each stage, and the final percentage of the original probes that were kept.

---

## Notebook 3 — Logistic Regression

**File:** `03_logistic_regression.ipynb`  
**Input:** `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`

### How Logistic Regression works

Logistic Regression is a linear classifier that models the probability that a sample belongs to a given class. Despite the name, it is a classification algorithm, not a regression one.

For each sample, the model computes a weighted sum of its features:

$$z = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b$$

This raw score $z$ is passed through the **sigmoid function**, which squashes it to a value between 0 and 1:

$$P(\text{HCC}) = \frac{1}{1 + e^{-z}}$$

If $P(\text{HCC}) \geq 0.5$, the sample is predicted as HCC; otherwise it is predicted as normal.

**Regularisation**  
With thousands of gene features and only a few hundred samples, Logistic Regression without regularisation would overfit badly — it would find weights that perfectly explain the training data but fail on new samples. Regularisation solves this by adding a penalty to the loss function that discourages large weights.

Two regularisation types are tested:
- **L2 (Ridge):** Penalises the sum of squared weights. This shrinks all weights toward zero smoothly but keeps all features in the model.
- **L1 (Lasso):** Penalises the sum of absolute weights. This drives the weights of less important features all the way to zero, effectively performing feature selection. With many irrelevant probes, L1 can produce a sparser, more interpretable model.

The strength of regularisation is controlled by the hyperparameter `C`, which is the **inverse** of regularisation strength. A small `C` means strong regularisation (simpler model); a large `C` means weak regularisation (fits the training data more closely). Both `C` and the penalty type are tuned using GridSearchCV.

**Interpretability**  
One of Logistic Regression's main advantages for gene expression analysis is interpretability. The learned coefficient for each probe directly indicates how strongly that probe influences the prediction — a large positive coefficient pushes the model toward predicting HCC, and a large negative coefficient pushes it toward normal. Because the data is scaled in Notebook 2, these coefficients are on the same scale and can be compared directly. The top 20 probes by absolute coefficient are visualised as a horizontal bar chart, with bars coloured red for probes pushing toward HCC and blue for probes pushing toward normal.

### Hyperparameter tuning — GridSearchCV

GridSearchCV tests every combination of hyperparameter values in the grid and selects the combination that gives the best mean cross-validation score. For Logistic Regression, the search space is small enough that an exhaustive grid search is practical.

**Search space:**

| Hyperparameter | Values | Description |
|---|---|---|
| `C` | 0.001, 0.01, 0.1, 1, 10, 100 | Inverse regularisation strength |
| `penalty` | `l1`, `l2` | Type of regularisation |
| `solver` | `liblinear` | Must use `liblinear` to support both L1 and L2 penalties |

**Total fits:** 6 × 2 × 1 = 12 combinations × 5 cross-validation folds = 60 model fits

All scoring uses F1 as the metric.

### Evaluation

After tuning, the best model is evaluated on the held-out test set:

- **Classification report:** Precision, recall, F1 score, and support for each class (HCC and normal), plus macro and weighted averages
- **Confusion matrix:** A 2×2 grid showing True Positives (HCC predicted as HCC), True Negatives (normal predicted as normal), False Positives (normal predicted as HCC), and False Negatives (HCC predicted as normal) — plotted with a blue colour scheme
- **ROC curve:** The trade-off between true positive rate (sensitivity) and false positive rate (1 − specificity) as the classification threshold is varied — the area under this curve (AUC) summarises overall discriminative ability; a value of 1.0 is perfect, 0.5 is no better than random
- **Top 20 feature coefficients:** A horizontal bar chart showing the 20 probes with the largest absolute coefficient values, coloured by direction

---

## Notebook 4 — Gradient Boosting and Model Comparison

**File:** `04_gradient_boosting.ipynb`  
**Input:** `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`

### How Gradient Boosting works

Gradient Boosting is an **ensemble method** that builds a strong predictive model by combining many weak models (shallow decision trees) in sequence. The key insight is that each new tree is trained to correct the errors of all previous trees.

The algorithm works as follows:

1. Start with a simple prediction — for example, always predict the majority class.
2. Calculate the **residuals**: how wrong were those predictions, and in which direction?
3. Train a new shallow decision tree to predict those residuals.
4. Add the new tree's predictions to the running total, scaled down by the **learning rate** to avoid overcorrecting.
5. Repeat steps 2–4 for a fixed number of iterations (`n_estimators`).

After many rounds, the ensemble gradually reduces its errors and converges on accurate predictions. Because each tree specifically targets the mistakes of the previous ones, the model tends to be more accurate than a single deep tree or even a Random Forest of the same size.

**Why it suits gene expression data**  
Gene expression features often have complex, non-linear relationships — a probe may only be informative in combination with another probe, or its effect may change depending on the value of a third probe. Decision trees capture these interaction effects naturally, and Gradient Boosting amplifies this by focusing each tree on the samples the previous trees got wrong.

**How it differs from XGBoost**  
XGBoost is also a gradient boosting algorithm and is what was used in the companion Sepsis project. Sklearn's `GradientBoostingClassifier` uses the original Friedman (2001) algorithm with exact greedy split finding at each node. XGBoost extends this with L1/L2 regularisation on the tree weights, an approximate split-finding algorithm for faster training on large datasets, and built-in handling of missing values. For a small dataset like this one (357 samples), both implementations perform comparably.

### Key hyperparameters

| Parameter | What it controls |
|---|---|
| `n_estimators` | Number of trees to build. More trees = more capacity, but the model takes longer to train and can overfit if pushed too far. |
| `learning_rate` | How much each new tree contributes to the final prediction. Smaller values mean each tree makes a more cautious correction — more trees are needed, but the model is less likely to overfit. |
| `max_depth` | How deep each individual tree can grow. Deeper trees can model more complex patterns but are also more prone to memorising the training data. |
| `subsample` | The fraction of training samples randomly selected to train each tree. Values below 1.0 introduce randomness that acts as a regulariser and can improve generalisation. |

### Hyperparameter tuning — Bayesian Optimisation

While Notebook 3 used GridSearchCV (exhaustive search over a fixed grid), Notebook 4 uses **Bayesian Optimisation** via `BayesSearchCV` from the `scikit-optimize` library.

GridSearchCV is effective when the search space is small and discrete, as was the case for Logistic Regression with 12 combinations. Gradient Boosting has four continuous hyperparameters that each span a wide range. A full grid search across all combinations would require hundreds or thousands of model fits. Bayesian optimisation solves this by being systematic rather than exhaustive:

1. It runs a small number of initial random trials to gather early information about the search space.
2. It builds a **surrogate model** — a probabilistic estimate of how performance varies across the search space, based on the results so far.
3. It uses the surrogate model to decide which combination of hyperparameters to try next, focusing on regions that look most promising.
4. After each new trial, it updates the surrogate model and repeats.

This means Bayesian optimisation gets progressively smarter with each trial, rather than sampling blindly. It typically finds better results than random search in the same number of trials.

**Search space:**

| Hyperparameter | Range | Scale | Description |
|---|---|---|---|
| `n_estimators` | 50 – 300 | Integer | Number of trees |
| `learning_rate` | 0.01 – 0.3 | Log-uniform | Step size per tree |
| `max_depth` | 2 – 6 | Integer | Tree depth |
| `subsample` | 0.5 – 1.0 | Uniform | Row sampling fraction per tree |

Log-uniform means values are sampled evenly on a log scale. This is appropriate for `learning_rate` because the difference between 0.01 and 0.05 is just as meaningful as the difference between 0.1 and 0.5 — both represent a 5x change in scale.

**Total fits:** 20 Bayesian trials × 5 cross-validation folds = 100 model fits

**Installation:** `scikit-optimize` is not pre-installed in Google Colab. The notebook includes a `!pip install scikit-optimize -q` cell that handles this automatically before the imports.

### Evaluation

The same evaluation suite as Notebook 3:
- Classification report
- Confusion matrix (orange colour scheme)
- ROC curve (orange)
- Top 20 feature importances — Gradient Boosting tracks how much each probe reduces prediction error across all trees. Unlike Logistic Regression coefficients, feature importances are always positive — they measure how much a probe contributes, not which direction it pushes the prediction.

### Model Comparison — Logistic Regression vs Gradient Boosting

The final section of Notebook 4 compares both models side by side. Logistic Regression is re-fitted using the best parameters found in Notebook 3 (which the `LR_BEST_PARAMS` variable at the top of the comparison cell). Using the same `X_train.csv` and `X_test.csv` files guarantees a fair comparison — both models see exactly the same data.

Four metrics are compared:

| Metric | What it measures |
|---|---|
| F1 Score | Harmonic mean of precision and recall for the HCC class — the primary comparison metric |
| ROC-AUC | Area under the ROC curve — measures overall discriminative ability regardless of threshold |
| Precision | Of all samples predicted as HCC, how many actually were HCC |
| Recall | Of all actual HCC samples, how many were correctly identified |

The comparison is shown as:
1. A printed table with all four metrics for both models
2. A grouped bar chart with blue bars for Logistic Regression and orange bars for Gradient Boosting, with the score labelled above each bar
3. Overlaid ROC curves for both models on the same plot, with the winner declared by F1 score in the final summary cell

---

## Key Design Decisions

### Why these two models?

**Logistic Regression** was chosen because it is the standard linear baseline for binary classification and is directly interpretable — the coefficient for each gene probe tells you its direction and strength of influence on the prediction. It also introduces students to regularisation (L1 vs L2, the `C` parameter), which is a core concept in machine learning. For scaled gene expression data, it often performs surprisingly well.

**Gradient Boosting** was chosen as the non-linear counterpart. It captures gene interaction effects that a linear model cannot, produces feature importances from a completely different mechanism than Logistic Regression coefficients, and introduces students to sequential ensemble methods and Bayesian hyperparameter search — both of which are more sophisticated techniques than the grid search used for Logistic Regression.

Together, the two models illustrate a key comparison that comes up constantly in applied machine learning: a simple, interpretable linear model versus a more complex, flexible ensemble. Sometimes the simpler model wins.

### Why four feature reduction steps and not just RFE?

RFE on its own would work, but it is slow when starting from 22,277 features. Each step in the pipeline dramatically reduces the number of features before the next step runs. By the time RFE runs, the feature set is already much smaller, making the sweep practical in a classroom setting.

Each step also has a different justification:
- Zero-variance removal eliminates features that are literally useless by definition.
- The high-null filter removes features that are too incomplete to be reliable.
- VarianceThreshold removes features that are nearly constant and would contribute almost nothing to any model.
- RFE then selects the best subset from what remains, using actual model performance as the selection criterion rather than a simple statistical threshold.

### Why is the same train/test split used for both model notebooks?

Both Notebook 3 and Notebook 4 read from the same `X_train.csv`, `X_test.csv`, `y_train.csv`, and `y_test.csv` files produced by Notebook 2. This guarantees that the comparison in Notebook 4 is fair — any difference in performance is due to the model, not a difference in which samples ended up in the training or test set.

### Why F1 score and not accuracy?

Although the classes are nearly balanced (50.7% vs 49.3%), F1 score is still preferred as the primary metric because it separately accounts for precision and recall. In a medical context, a False Negative (predicting a cancer sample as normal) and a False Positive (predicting a normal sample as cancer) have very different clinical consequences. F1 makes both types of error visible, while accuracy treats them the same.

---

## Libraries Used

| Library | Version | Purpose |
|---|---|---|
| `pandas` | any recent | Data loading, manipulation, and saving |
| `numpy` | any recent | Numerical operations and array handling |
| `matplotlib` | any recent | All plots and visualisations |
| `seaborn` | any recent | Plot styling and box plots |
| `scikit-learn` | ≥ 1.0 | Logistic Regression, Gradient Boosting, preprocessing, feature selection, metrics |
| `scikit-optimize` | ≥ 0.9 | `BayesSearchCV` for Bayesian hyperparameter search (Notebook 4 only) |

All libraries except `scikit-optimize` are pre-installed in Google Colab. `scikit-optimize` is installed automatically by a cell at the start of the tuning section in Notebook 4.

---

## Results Summary

### Feature Reduction (Notebook 2)

| Stage | Features Remaining | Removed |
|---|---|---|
| Original gene probes | 22,277 | — |
| After zero-variance removal | 22,277 | 0 |
| After high-null removal (>90%) | 22,277 | 0 |
| After VarianceThreshold (0.01) | 22,277 | 0 |
| After RFE (best k = 30) | 30 | 22,247 |

This dataset is already clean and fully normalised, so the first three steps remove nothing — all probes have non-zero variance, no missing values, and no near-constant features. RFE does all the work, reducing from 22,277 probes down to 30. These 30 probes represent 0.13% of the original feature space.

**30 probes selected by RFE:** `201268_at`, `201293_x_at`, `202544_at`, `202824_s_at`, `202868_s_at`, `202983_at`, `203316_s_at`, `204428_s_at`, `204641_at`, `205307_s_at`, `205554_s_at`, `206938_at`, `207407_x_at`, `207584_at`, `207608_x_at`, `207609_s_at`, `207995_s_at`, `208491_at`, `209365_s_at`, `209614_at`, `209714_s_at`, `209976_s_at`, `210481_s_at`, `211295_x_at`, `213629_x_at`, `214320_x_at`, `214677_x_at`, `216661_x_at`, `217022_s_at`, `217546_at`

### SMOTE (Notebook 2)

| | HCC | Normal | Total |
|---|---|---|---|
| Before SMOTE | 144 | 141 | 285 |
| After SMOTE | 500 | 500 | 1,000 |

The training set grows from 285 to 1,000 samples. The test set (72 samples: 37 HCC, 35 Normal) is never touched by SMOTE.

### Model Performance (Notebooks 3 and 4)

| Model | F1 Score | ROC-AUC | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.9118 | 0.9745 | 0.85 | 1.00 |
| Logistic Regression (tuned, GridSearch) | 0.8308 | 0.9081 | 0.9643 | 0.7297 |
| Gradient Boosting (baseline) | 0.9722 | 0.9660 | 0.95 | 1.00 |
| Gradient Boosting (tuned, BayesSearch) | **0.9722** | **0.9753** | **1.0000** | **0.9459** |

Gradient Boosting outperforms Logistic Regression on every metric. The tuned Gradient Boosting model achieves perfect precision (no false positives) while catching 94.6% of all HCC cases in the test set.

### Confusion Matrices

**Logistic Regression (tuned):** TP=27, TN=34, FP=1, FN=10  
**Gradient Boosting (tuned):** TP=35, TN=35, FP=0, FN=2

The Gradient Boosting model misclassified only 2 samples out of 72 — both were HCC cases predicted as Normal. It made zero false positive errors, meaning every sample it flagged as HCC was genuinely cancerous.

### Best Hyperparameters

**Logistic Regression (GridSearchCV):** `C=100`, `penalty='l2'`, `solver='liblinear'`

**Gradient Boosting (BayesSearchCV):** `n_estimators=174`, `learning_rate=0.0635`, `max_depth=6`, `subsample=0.9188`

### Most Influential Features

**Logistic Regression — top coefficients (by absolute value):**

| Probe | Coefficient | Direction |
|---|---|---|
| `204428_s_at` | −15.45 | toward Normal |
| `216661_x_at` | +10.32 | toward HCC |
| `209614_at` | +9.61 | toward HCC |
| `209365_s_at` | +9.55 | toward HCC |
| `204641_at` | −8.83 | toward Normal |

**Gradient Boosting — top feature importances:**

| Probe | Importance |
|---|---|
| `209365_s_at` | 0.8324 |
| `216661_x_at` | 0.0280 |
| `202544_at` | 0.0257 |
| `207584_at` | 0.0148 |
| `205307_s_at` | 0.0133 |

The probe `209365_s_at` dominates the Gradient Boosting model with an importance of 0.83 — it alone accounts for 83% of the total split criterion reduction across all trees. This probe corresponds to the gene **GPC3 (Glypican-3)**, a well-established biomarker for hepatocellular carcinoma that is overexpressed in HCC tissue and nearly absent in normal adult liver.

---

## Interpretation

These results are strong, and that reflects the nature of the dataset. GSE14520 is a clean, well-curated microarray study with a clear biological signal — HCC tumour tissue versus non-tumour liver tissue from the same patients. The gene expression differences between cancer and healthy tissue are substantial and consistent, which makes classification relatively tractable once the right features are selected.

**Why Gradient Boosting outperforms Logistic Regression here:**  
Logistic Regression is a linear model — it can only separate classes using a straight hyperplane in the feature space. Gene expression data often has non-linear relationships and interaction effects between probes. Gradient Boosting's sequential decision trees capture these interactions directly, giving it an edge. The gap is especially visible in recall: Logistic Regression misses 10 HCC cases (27%) while Gradient Boosting misses only 2 (5.4%).

**Why the tuned Logistic Regression scores lower than baseline:**  
The baseline Logistic Regression (C=1, L2) happens to be well-calibrated for this data. The tuned model (C=100, L2) uses much weaker regularisation, which allows it to fit the SMOTE-augmented training set very tightly — a sign of mild overfitting on the synthetic samples. The CV F1 during GridSearch (0.9889) is much higher than the test F1 (0.8308), which confirms the overfitting. This is a useful teaching moment: higher CV score during tuning does not always translate to better test performance.


