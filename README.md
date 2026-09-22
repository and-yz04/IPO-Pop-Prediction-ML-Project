# Predicting IPO First-Day "Pops"

A supervised classification project predicting whether a U.S. IPO will "pop" (close its first trading day above the offer price) using only information available before the stock starts trading.

## Overview

When a company IPOs, its shares are offered at a set price before trading opens. If the stock's price jumps above that offer price on day one, it's said to have "popped" — a sign the offering was underpriced. This project builds a model to predict that outcome using firm characteristics and market conditions known *before* trading begins, with an eye toward the kind of pre-listing risk assessment relevant to broker-dealers and the firms that insure them.

## Data

- **Source:** [Stocks IPO information & results](https://www.kaggle.com/datasets/proselotis/financial-ipo-data), Kaggle dataset by `proselotis`
- **Scope:** 3,762 U.S. IPOs listed between 1996 and 2018
- **Features used:** sector, company age, employee count, U.S.-based status, and market trend in the months before listing

## Methods

- Data cleaning and feature engineering in R (`tidyverse`, `janitor`)
- Exploratory analysis of pop rates by sector, company age, employee count, and market trend
- Stratified train/test split with 10-fold cross-validation on the training set
- Class imbalance addressed via upsampling (`themis::step_upsample`)
- Five classification models trained and tuned on identical folds/recipe:
  - Logistic regression (baseline)
  - Linear discriminant analysis (baseline)
  - Elastic net logistic regression
  - Random forest
  - Boosted trees
- Model comparison via cross-validated ROC AUC (chosen over accuracy due to class imbalance)

## Results

| Model | CV ROC AUC |
|---|---|
| **Random forest (selected)** | **0.710** |
| Boosted tree | 0.703 |
| Elastic net | 0.693 |
| LDA | 0.691 |
| Logistic regression | 0.689 |

The final random forest scored **0.696 ROC AUC** and **76.5% accuracy** on the held-out test set, consistent with its cross-validated performance. Variable importance and a confusion matrix for the final model are included in the report.

## Tools

R, tidymodels, tidyverse, janitor, corrplot, discrim, vip, kknn, themis, doParallel

## Files

- `ipo-pop-final-project.Rmd` — full analysis: EDA, recipe, model fitting/tuning, evaluation
- `ipo-pop-final-project.html` — knitted report (open locally, or view via [htmlpreview](https://htmlpreview.github.io/) once pushed)

## Author

Andy — UC Santa Barbara, Financial Mathematics & Statistics
