import joblib
import pandas as pd

# load models
kmeans = joblib.load('model/kmeans_model.pkl')
preprocessor = joblib.load('model/preprocessor.pkl')

FEATURES = ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]

def predict_clusters(customer_features):
    """
    customer_features must be a list:
    [Gender, Age, Annual Income, Spending Score]
    """
    df = pd.DataFrame([customer_features], columns=FEATURES)

    # preprocess with pipeline
    X_scaled = preprocessor.transform(df)

    # predict cluster
    cluster_label = int(kmeans.predict(X_scaled)[0])

    cluster_names = {
        0: "VIP Customers",
        1: "Occasional Buyers",
        2: "At-Risk Customers",
        3: "New Subscribers"
    }

    return {
        "cluster": cluster_label,
        "segment_name": cluster_names.get(cluster_label, "Unknown")
    }
