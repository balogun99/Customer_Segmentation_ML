import streamlit as st
import pandas as pd
from src.utils import predict_clusters
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Segmentation", layout="centered")
st.title("Customer Segmentation for Marketing Strategy")

# upload CSV
uploaded_file = st.file_uploader(
    "Upload Customer CSV (with Gender, Age, Annual Income, Spending Score)", type=['csv']
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Customer Data Preview")
    st.dataframe(df.head())

    # load models
    preprocessor = joblib.load('model/preprocessor.pkl')
    kmeans = joblib.load('model/kmeans_model.pkl')

    numeric_features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    categorical_features = ["Gender"]
    FEATURES = categorical_features + numeric_features

    # preprocess
    X_scaled = preprocessor.transform(df[FEATURES])

    # PCA visualization on numeric part only
    pca = PCA(n_components=2)
    X_numeric_scaled = X_scaled[:, :len(numeric_features)]
    pca_result = pca.fit_transform(X_numeric_scaled)

    clusters = kmeans.predict(X_scaled)
    df['cluster'] = clusters
    df['segment'] = df['cluster'].map({
        0: "VIP Customers",
        1: "Occasional Buyers",
        2: "At-Risk Customers",
        3: "New Subscribers"
    })

    # plot clusters
    fig, ax = plt.subplots()
    for label in df['cluster'].unique():
        ax.scatter(
            pca_result[df['cluster'] == label, 0],
            pca_result[df['cluster'] == label, 1],
            label=df['segment'][df['cluster'] == label].iloc[0]
        )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Customer Segments (PCA Projection)")
    ax.legend()
    st.pyplot(fig)

# single customer prediction
st.subheader("Predict Segment for Single Customer")
Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.number_input("Age", min_value=0, value=30)
Yearly = st.number_input("Annual Income (k$)", min_value=0, value=60)
Expenses = st.number_input("Spending Score (1-100)", min_value=0, value=50)

if st.button("Predict Segment"):
    result = predict_clusters([Gender, Age, Yearly, Expenses])
    st.write(f"Cluster: **{result['cluster']}**")
    st.write(f"Segment: **{result['segment_name']}**")
