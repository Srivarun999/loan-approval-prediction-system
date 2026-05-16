import pandas as pd
import numpy as np

print('Loading dataset...')
df = pd.read_csv('loan_approval_dataset.csv')
# Normalize same as app
_df = df.copy()
_df.columns = _df.columns.str.strip().str.lower().str.replace(' ', '_')
for col in _df.select_dtypes(include='object').columns:
    _df[col] = _df[col].str.strip()

print('\nColumns:')
print(list(_df.columns))

# Map common names
rename_map = {
    'applicant_id':'loan_id', 'annual_income':'income_annum', 'credit_score':'cibil_score'
}
_df.rename(columns={k:v for k,v in rename_map.items() if k in _df.columns}, inplace=True)

features = ['no_of_dependents','education','self_employed','income_annum','loan_amount','loan_term','cibil_score',
            'residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value','loan_status']

print('\nFeature presence and unique counts:')
for f in features:
    if f in _df.columns:
        print(f"- {f}: present; nunique={_df[f].nunique()}; top_values=\n{_df[f].value_counts().head(5)}\n")
    else:
        print(f"- {f}: MISSING")

# Numeric summary for numeric features present
num_feats = [f for f in ['income_annum','loan_amount','cibil_score','loan_term','no_of_dependents'] if f in _df.columns]
if num_feats:
    print('\nNumeric summary:')
    print(_df[num_feats].describe().T)

# Check for near-constant columns
print('\nColumns with low variance (unique <= 2):')
for c in _df.columns:
    if _df[c].nunique() <= 2:
        print(f"- {c}: nunique={_df[c].nunique()} | top=\n{_df[c].value_counts().head()}\n")

# Correlation with target if loan_status present
if 'loan_status' in _df.columns:
    print('\nCorrelations with target (for numeric features):')
    # encode target binary
    try:
        t = _df['loan_status'].astype('category').cat.codes
        for f in num_feats:
            corr = _df[f].astype(float).corr(t)
            print(f"- {f} vs loan_status: corr={corr:.3f}")
    except Exception as e:
        print('Could not compute correlations:', e)

print('\nDone')
