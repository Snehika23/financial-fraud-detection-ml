import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('best_model.pkl')
feature_cols = joblib.load('feature_cols.pkl')

st.set_page_config(page_title="Fraud Detector", layout="wide")
st.title("Credit Card Fraud Detection System")
st.write("This system analyses credit card transactions and flags potentially fraudulent activity.")

st.sidebar.header("Select a Transaction")

@st.cache_data
def load_data():
    df = pd.read_csv('creditcard.csv')
    from sklearn.preprocessing import StandardScaler
    df['Amount_Scaled'] = StandardScaler().fit_transform(df[['Amount']])
    df['Time_Scaled'] = StandardScaler().fit_transform(df[['Time']])
    df['Original_Amount'] = df['Amount']
    df = df.drop(['Amount', 'Time'], axis=1)
    return df

df = load_data()

mode = st.sidebar.radio("Mode", ["Random Legitimate", "Random Fraud", "Browse Transactions"])

if mode == "Browse Transactions":
    # Build readable labels
    sample_indices = list(range(0, min(500, len(df))))
    labels = {i: f"TXN-{i:06d}  |  ${df.iloc[i]['Original_Amount']:.2f}  |  {'FRAUD' if df.iloc[i]['Class']==1 else 'Legit'}" for i in sample_indices}
    selected = st.sidebar.selectbox("Choose a transaction", sample_indices, format_func=lambda x: labels[x])
    idx = selected

elif mode == "Random Fraud":
    if st.sidebar.button("Generate Fraud Sample", type="primary"):
        st.session_state['idx'] = df[df['Class'] == 1].sample(1).index[0]
    idx = st.session_state.get('idx', None)

else:
    if st.sidebar.button("Generate Legitimate Sample", type="primary"):
        st.session_state['idx'] = df[df['Class'] == 0].sample(1).index[0]
    idx = st.session_state.get('idx', None)

if idx is not None:
    row = df.iloc[idx]
    input_df = row[feature_cols].to_frame().T

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    actual = int(row['Class'])

    st.subheader("Analysis Result")
    col1, col2, col3 = st.columns(3)

    with col1:
        if prediction == 1:
            st.error("FRAUD DETECTED", icon="🚨")
        else:
            st.success("Transaction Legitimate", icon="✅")

    with col2:
        st.metric("Fraud Probability", f"{probability[1]*100:.2f}%")
        st.metric("Legitimate Probability", f"{probability[0]*100:.2f}%")

    with col3:
        st.metric("Transaction Amount", f"${row['Original_Amount']:.2f}")
        if actual == 1:
            st.warning("Actual: FRAUD")
        else:
            st.info("Actual: Legitimate")

    st.subheader("Top Features for this Transaction")
    top_features = input_df.T.rename(columns={input_df.index[0]: 'Value'}).round(4)
    st.dataframe(top_features)

st.markdown("---")
st.caption("COM 763 — Advanced Machine Learning | Credit Card Fraud Detection System")