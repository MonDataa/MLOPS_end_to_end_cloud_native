from datetime import timedelta

from feast import FeatureView, Field, FileSource
from feast.types import Float64, Int64, String

from entities import customer


FEATURE_PATH = '/shared/lake/exports/feast/credit_risk_features.parquet'

credit_risk_source = FileSource(
    path=FEATURE_PATH,
    event_timestamp_column='event_timestamp',
)

credit_risk_feature_view = FeatureView(
    name='credit_risk_features',
    entities=[customer],
    ttl=timedelta(days=30),
    schema=[
        Field(name='application_id', dtype=Int64),
        Field(name='bureau_score', dtype=Float64),
        Field(name='open_accounts', dtype=Int64),
        Field(name='delinquencies_2y', dtype=Int64),
        Field(name='inquiries_6m', dtype=Int64),
        Field(name='revolving_utilization', dtype=Float64),
        Field(name='debt_to_income', dtype=Float64),
        Field(name='annual_income', dtype=Float64),
        Field(name='years_employed', dtype=Float64),
        Field(name='loan_amount', dtype=Float64),
        Field(name='loan_term_months', dtype=Int64),
        Field(name='interest_rate', dtype=Float64),
        Field(name='requested_payment', dtype=Float64),
        Field(name='loan_to_income', dtype=Float64),
        Field(name='installment_to_income', dtype=Float64),
        Field(name='payments_late_12m', dtype=Int64),
        Field(name='late_payment_rate_12m', dtype=Float64),
        Field(name='months_since_last_late', dtype=Int64),
        Field(name='previous_defaults', dtype=Int64),
        Field(name='credit_history_risk_score', dtype=Float64),
        Field(name='employment_status', dtype=String),
        Field(name='housing_status', dtype=String),
        Field(name='purpose', dtype=String),
        Field(name='gender', dtype=String),
        Field(name='age_group', dtype=String),
    ],
    online=True,
    source=credit_risk_source,
)
