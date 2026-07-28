# app.py -- Simple Streamlit AI Analysis App (Imran)
import re, os, joblib, uuid
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt

# Config
MAX_LAG = 10
RANDOM_STATE = 42
MODEL_DIR = "models_streamlit"
os.makedirs(MODEL_DIR, exist_ok=True)

# Helpers
def parse_line(line):
    line = line.strip()
    if not line:
        return None
    parts = line.split()
    id_str = parts[0]
    token = ''.join(parts[1:]) if len(parts) >= 2 else ''
    m = re.match(r'(\d+)([A-Za-z]*)', token)
    if not m:
        return None
    value = int(m.group(1))
    size = m.group(2).capitalize() if m.group(2) else ''
    return {'id': id_str, 'value': value, 'size': size}

def id_to_datetime(id_str):
    s = id_str
    if len(s) < 14:
        return None
    base = s[:14]
    try:
        dt = datetime.strptime(base, '%Y%m%d%H%M%S')
    except:
        return None
    if len(s) > 14:
        ms = s[14:]
        try:
            ms_int = int(ms[:3].ljust(3, '0'))
            dt = dt + timedelta(milliseconds=ms_int)
        except:
            pass
    return dt

def load_raw_from_text(text):
    rows = []
    for line in text.splitlines():
        p = parse_line(line)
        if p:
            p['dt'] = id_to_datetime(p['id'])
            rows.append(p)
    df = pd.DataFrame(rows)
    if 'dt' in df.columns and df['dt'].notnull().all():
        df = df.sort_values('dt').reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    df['size'] = df['size'].replace('', np.nan)
    df['size'] = df['size'].fillna(method='ffill').fillna('Small')
    return df

def make_features(df):
    df = df.copy()
    df['size_enc'] = df['size'].map({'Small':0, 'Big':1}).fillna(0).astype(int)
    for lag in range(1, MAX_LAG+1):
        df[f'lag_{lag}'] = df['value'].shift(lag)
    df['roll_mean_3'] = df['value'].rolling(3).mean().shift(1)
    df['roll_std_3'] = df['value'].rolling(3).std().shift(1).fillna(0)
    df['target_value_next'] = df['value'].shift(-1)
    df['target_size_next'] = df['size'].shift(-1)
    df_model = df.dropna(subset=[f'lag_{MAX_LAG}', 'target_value_next', 'target_size_next']).reset_index(drop=True)
    return df_model

def train_models(df_model):
    feature_cols = [f'lag_{i}' for i in range(1, MAX_LAG+1)] + ['roll_mean_3', 'roll_std_3', 'size_enc']
    X = df_model[feature_cols].astype(float)
    y_val = df_model['target_value_next'].astype(int)
    y_size = (df_model['target_size_next'].map({'Small':0, 'Big':1}).astype(int))
    split_idx = int(0.8 * len(X))
    if split_idx < 10:
        split_idx = int(0.7 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_val_train, y_val_test = y_val.iloc[:split_idx], y_val.iloc[split_idx:]
    y_size_train, y_size_test = y_size.iloc[:split_idx], y_size.iloc[split_idx:]

    digit_model = xgb.XGBClassifier(
        objective='multi:softprob', num_class=10,
        n_estimators=200, max_depth=4, learning_rate=0.1,
        use_label_encoder=False, random_state=RANDOM_STATE, eval_metric='mlogloss'
    )
    digit_model.fit(X_train, y_val_train)
    size_model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=200, max_depth=4, learning_rate=0.1,
        use_label_encoder=False, random_state=RANDOM_STATE, eval_metric='logloss'
    )
    size_model.fit(X_train, y_size_train)

    val_pred = digit_model.predict(X_test)
    size_pred = size_model.predict(X_test)
    acc_val = accuracy_score(y_val_test, val_pred) if len(y_val_test)>0 else 0.0
    acc_size = accuracy_score(y_size_test, size_pred) if len(y_size_test)>0 else 0.0

    joblib.dump({'model': digit_model, 'features': feature_cols}, os.path.join(MODEL_DIR, 'digit_model.joblib'))
    joblib.dump({'model': size_model, 'features': feature_cols}, os.path.join(MODEL_DIR, 'size_model.joblib'))

    return {'digit_acc': acc_val, 'size_acc': acc_size, 'feature_cols': feature_cols}

def load_models():
    digit_path = os.path.join(MODEL_DIR, 'digit_model.joblib')
    size_path = os.path.join(MODEL_DIR, 'size_model.joblib')
    if os.path.exists(digit_path) and os.path.exists(size_path):
        d = joblib.load(digit_path)
        s = joblib.load(size_path)
        return d['model'], s['model'], d['features']
    return None, None, None

def build_next_features(df_tail, feature_cols):
    last = df_tail.tail(MAX_LAG+1).reset_index(drop=True)
    feats = {}
    for lag in range(1, MAX_LAG+1):
        feats[f'lag_{lag}'] = last['value'].iloc[-lag]
    feats['roll_mean_3'] = last['value'].iloc[-3:].mean()
    feats['roll_std_3'] = last['value'].iloc[-3:].std() if last['value'].iloc[-3:].std()==last['value'].iloc[-3:].std() else 0.0
    feats['size_enc'] = 1 if str(last['size'].iloc[-1]).strip().lower() == 'big' else 0
    Xnext = pd.DataFrame([feats])[feature_cols].astype(float)
    return Xnext

# Streamlit UI
st.set_page_config(page_title="Simple AI Analysis App", layout="centered")
st.title("Simple AI Analysis App — Imran")
st.markdown("Upload raw file, Train models, then Predict next number (0-9) and size (Small/Big).")

uploaded = st.file_uploader("Upload raw text file (each line like: 20260728100030095 1 Small OR 20260728100030094 9Big)", type=["txt","csv"])
if uploaded is not None:
    raw_text = uploaded.read().decode('utf-8')
    df = load_raw_from_text(raw_text)
    st.subheader("Data preview (first 10 rows)")
    st.dataframe(df.head(10))
    st.write(f"Total rows: {len(df)}")
    if len(df) < MAX_LAG + 5:
        st.warning(f"Too few rows for reliable training. Need at least {MAX_LAG+5} rows. You have {len(df)}.")
    st.subheader("Value distribution (last 200)")
    vals = df['value'].tail(200).value_counts().sort_index()
    st.bar_chart(vals)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Train models on this file"):
            with st.spinner("Training..."):
                dfm = make_features(df)
                if len(dfm) < 20:
                    st.error("Not enough prepared rows after features. Need >= ~20.")
                else:
                    res = train_models(dfm)
                    st.success("Training complete.")
                    st.write(f"Digit model test accuracy: {res['digit_acc']:.3f}")
                    st.write(f"Size model test accuracy: {res['size_acc']:.3f}")
    with col2:
        if st.button("Predict next (using uploaded file)"):
            digit_model, size_model, feat_cols = load_models()
            if digit_model is None:
                st.error("No trained models found. First press 'Train models on this file'.")
            else:
                if len(df) < MAX_LAG + 2:
                    st.error(f"Not enough rows to build features. Need at least {MAX_LAG+2} rows.")
                else:
                    Xnext = build_next_features(df, feat_cols)
                    proba_digit = digit_model.predict_proba(Xnext)[0]
                    pred_digit = int(digit_model.predict(Xnext)[0])
                    proba_size = size_model.predict_proba(Xnext)[0]
                    pred_size = "Big" if int(size_model.predict(Xnext)[0])==1 else "Small"
                    st.write("Predicted next digit:", pred_digit)
                    st.write("Digit probabilities (0-9):")
                    probs = {i: float(proba_digit[i]) for i in range(len(proba_digit))}
                    st.json(probs)
                    st.write("Predicted next size:", pred_size)
                    st.write("Size probabilities:")
                    st.json({"Small": float(proba_size[0]), "Big": float(proba_size[1])})
                    fig, ax = plt.subplots(figsize=(6,2.5))
                    tail = df['value'].tail(50).reset_index(drop=True)
                    ax.plot(tail.index, tail.values, marker='o')
                    ax.set_title("Last values (tail)")
                    ax.set_ylabel("value")
                    st.pyplot(fig)
else:
    st.info("Upload a raw data file to start (see sample format).")
