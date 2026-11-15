import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

# G, C, B, A are 1/0 flags (ground, phase C, phase B, phase A) -> fault type name
LABELS = {
    "0000": "No Fault",
    "1001": "LG (A-G)",
    "0110": "LL (B-C)",
    "1011": "LLG (A-B-G)",
    "0111": "LLL (A-B-C)",
    "1111": "LLLG (A-B-C-G)",
}

def load_data(path="classData.csv"):
    df = pd.read_csv(path)
    code = df["G"].astype(str) + df["C"].astype(str) + df["B"].astype(str) + df["A"].astype(str)
    df["fault"] = code.map(LABELS)
    return df

def split_and_scale(df, test_size=0.2, seed=42):
    X, y = df[FEATURES], df["fault"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    scaler = StandardScaler().fit(X_tr)   # fit on train only
    return scaler.transform(X_tr), scaler.transform(X_te), y_tr, y_te, scaler
