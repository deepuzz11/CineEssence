from sklearn.neighbors import NearestNeighbors
import pandas as pd
import joblib

# Load data
movie_ratings = pd.read_csv('cleaned_movie_ratings.csv')

# Create a pivot table for user-item interactions
user_movie_matrix = movie_ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Fit KNN model
model = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
model.fit(user_movie_matrix)

# Example: Generate recommendations for the first user
user_index = 0  # index of the user for whom to generate recommendations
distances, indices = model.kneighbors(user_movie_matrix.iloc[user_index, :].values.reshape(1, -1), n_neighbors=6)

# Get movieId of the nearest neighbors (excluding the user itself)
recommended_movie_indices = indices.flatten()[1:]

# Create a DataFrame for the recommendations
recommended_movies = user_movie_matrix.columns[recommended_movie_indices]
recommendations_df = pd.DataFrame({
    'userId': [user_movie_matrix.index[user_index]] * len(recommended_movies),
    'movieId': recommended_movies,
    'predicted_rating': [4.5] * len(recommended_movies)  # This is just an example rating
})

# Save recommendations to a CSV file
recommendations_df.to_csv('recommendations.csv', index=False)

# Save the model
joblib.dump(model, 'recommendation_model.pkl')
