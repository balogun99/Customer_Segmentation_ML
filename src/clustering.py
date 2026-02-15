import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from preprocessing import preprocess_data

# preprocess dataset
X_scaled, preprocessor, df = preprocess_data('data/Mall_Customers.csv')

# number of clusters
k = 4

# train KMeans
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# add cluster label to the dataframe# add cluster label to the dataframe
df['cluster'] = clusters

# silhouette score
score = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {score}")

# save the trained objects
joblib.dump(kmeans, 'model/kmeans_model.pkl')
joblib.dump(preprocessor, 'model/preprocessor.pkl')  # save pipeline
print("Successfully saved KMeans model and preprocessor")

# save processed dataframe
df.to_csv('data/processed/customer_clusters.csv', index=False)
