import pandas as pd

# Replicate the preprocessing logic from app.py (non-Streamlit)

def run_test():
    df = pd.read_csv("loan_approval_dataset.csv")
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    rename_map = {
        "applicant_id": "loan_id",
        "annual_income": "income_annum",
        "credit_score": "cibil_score",
        "employment_years": "no_of_dependents",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    required = ["loan_amount", "income_annum"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("MISSING:", missing)
        return

    # Compute loan_income_ratio
    df["loan_income_ratio"] = df["loan_amount"] / (df["income_annum"] + 1)
    print("Columns after normalization:", list(df.columns))
    print(df[["loan_amount","income_annum","loan_income_ratio" ]].head())

if __name__ == '__main__':
    run_test()
