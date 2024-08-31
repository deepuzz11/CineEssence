from sklearn.neighbors import NearestNeighbors
import pandas as pd

# Load data
movie_ratings = pd.read_csv('cleaned_movie_ratings.csv')

# Example: Build a simple recommendation model (e.g., KNN)
model = NearestNeighbors(n_neighbors=5)
model.fit(movie_ratings[['rating']])

# Save the model
import joblib
joblib.dump(model, 'recommendation_model.pkl')
