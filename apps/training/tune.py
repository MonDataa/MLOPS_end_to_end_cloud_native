"""Optional hyperparameter tuning for the credit-risk model using real Delta gold data."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from deltalake import DeltaTable
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


GOLD_PATH = Path(os.environ.get('GOLD_FEATURES_PATH', '/shared/lake/gold/credit_risk_features'))
REPORT_ROOT = Path(os.environ.get('MODEL_REPORT_ROOT', '/shared/model_reports'))
TARGET_COLUMN = 'defaulted'
NUMERIC_FEATURES = [
    'bureau_score',
    'open_accounts',
    'delinquencies_2y',
    'inquiries_6m',
    'revolving_utilization',
    'debt_to_income',
    'annual_income',
    'years_employed',
    'loan_amount',
    'loan_term_months',
    'interest_rate',
    'requested_payment',
    'loan_to_income',
    'installment_to_income',
    'payments_late_12m',
    'late_payment_rate_12m',
    'months_since_last_late',
    'previous_defaults',
    'credit_history_risk_score',
]
CATEGORICAL_FEATURES = ['employment_status', 'housing_status', 'purpose']
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), NUMERIC_FEATURES),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')),
        ]
    )


def main() -> None:
    if not GOLD_PATH.exists():
        raise FileNotFoundError('No credit-risk gold Delta table found; run feature engineering first.')

    dataset = DeltaTable(str(GOLD_PATH)).to_pandas().dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid={
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__penalty': ['l1', 'l2'],
        },
        scoring='roc_auc',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    )
    search.fit(dataset[FEATURE_COLUMNS], dataset[TARGET_COLUMN].astype(int))

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    output = {
        'best_params': search.best_params_,
        'best_roc_auc': float(search.best_score_),
    }
    (REPORT_ROOT / 'tuning_report.json').write_text(json.dumps(output, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
