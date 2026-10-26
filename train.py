import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, recall_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# 1. LOAD + CLEAN
df = pd.read_csv("data/churn.csv")
df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # blanks -> NaN
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Save raw defaults for the Streamlit app later
from pandas.api.types import is_numeric_dtype

raw_X = df.drop(columns=["Churn"])
num_cols = [c for c in raw_X.columns if is_numeric_dtype(raw_X[c])]
cat_cols = [c for c in raw_X.columns if c not in num_cols]

defaults = {c: raw_X[c].median() for c in num_cols}
defaults.update({c: raw_X[c].mode()[0] for c in cat_cols})

# 2. ENCODE
X = pd.get_dummies(raw_X, drop_first=True).astype(float)
y = df["Churn"]

# 3. SPLIT FIRST (held-out test set stays untouched)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 4. SMOTE on TRAIN ONLY
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

# 5. TRAIN 4 MODELS, COMPARE ON SAME TEST DATA
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05,
                             max_depth=4, eval_metric="logloss", random_state=42),
}
rows = []
for name, m in models.items():
    m.fit(X_res, y_res)
    pred = m.predict(X_test)
    proba = m.predict_proba(X_test)[:, 1]
    rows.append({"Model": name,
                 "F1": f1_score(y_test, pred),
                 "Recall": recall_score(y_test, pred),
                 "ROC-AUC": roc_auc_score(y_test, proba)})
results = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
print(results.round(3))
results.to_csv("models/results.csv", index=False)

# 6. SAVE XGBoost (SHAP works best on tree models) + helper info
joblib.dump(models["XGBoost"], "models/xgb.pkl")
joblib.dump(list(X.columns), "models/columns.pkl")
joblib.dump(defaults, "models/defaults.pkl")
joblib.dump({c: sorted(raw_X[c].unique()) for c in cat_cols}, "models/options.pkl")