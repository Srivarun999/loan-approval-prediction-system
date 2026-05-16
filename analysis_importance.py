import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and normalize
df = pd.read_csv('loan_approval_dataset.csv')
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# Map common names
rename_map = {
    'applicant_id':'loan_id', 'annual_income':'income_annum', 'credit_score':'cibil_score'
}
df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns}, inplace=True)

# Provide defaults for missing optional fields
defaults = {
    'self_employed':'No',
    'education':'Not Graduate',
    'loan_status':'Rejected',
    'no_of_dependents':0,
    'loan_term':10,
}
for col,val in defaults.items():
    if col not in df.columns:
        df[col] = val

# Ensure numeric columns
num_cols = ['no_of_dependents','income_annum','loan_amount','loan_term','cibil_score']
for nc in num_cols:
    if nc in df.columns:
        df[nc] = pd.to_numeric(df[nc], errors='coerce')

# Drop rows missing core values
if 'loan_amount' in df.columns and 'income_annum' in df.columns:
    df.dropna(subset=['loan_amount','income_annum','loan_status'], inplace=True)

# Feature engineering
if 'loan_amount' in df.columns and 'income_annum' in df.columns:
    df['loan_income_ratio'] = df['loan_amount'] / (df['income_annum'] + 1)

# Encode categorical columns
encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    try:
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    except Exception as e:
        pass

# Build feature list: numeric + *_enc except target
feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != 'loan_status_enc']
enc_cols = [c for c in df.columns if c.endswith('_enc') and c != 'loan_status_enc']
feature_cols = [c for c in feature_cols if c not in enc_cols] + enc_cols

print('Using features:', feature_cols)

# Ensure target
if 'loan_status_enc' not in df.columns:
    if 'loan_status' in df.columns:
        df['loan_status_enc'] = LabelEncoder().fit_transform(df['loan_status'])
    else:
        print('No target column found; aborting')
        raise SystemExit

X = df[feature_cols].fillna(0)
y = df['loan_status_enc']

# Train RandomForest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
print('Train acc:', accuracy_score(y_train, model.predict(X_train)))
print('Test acc :', accuracy_score(y_test, model.predict(X_test)))

# Feature importances
importances = model.feature_importances_
feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
print('\nTop features:')
for f, imp in feat_imp[:15]:
    print(f'{f}: {imp:.4f}')

# Show label mapping for loan_status
if 'loan_status' in df.columns:
    le = LabelEncoder().fit(df['loan_status'])
    print('\nLoan status classes:', list(le.classes_))

# Show a few example predictions where model disagrees with human-looking rules
print('\nExamples where model predicts Approved but CIBIL<600 and loan_income_ratio>5:')
if 'cibil_score' in df.columns and 'loan_income_ratio' in df.columns:
    cond = (df['cibil_score']<600)&(df['loan_income_ratio']>5)
    if cond.any():
        sample = df[cond].head(10)
        preds = model.predict(sample[feature_cols].fillna(0))
        sample = sample.copy()
        sample['model_pred'] = preds
        print(sample[['cibil_score','loan_income_ratio','loan_amount','income_annum','loan_status','model_pred']].head(10))

print('\nDone')
