import pandas as pd

# Load datasets
links_df = pd.read_csv('links.csv')
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')
tags_df = pd.read_csv('tags.csv')

# Example preprocessing: Merge ratings with movies to get movie titles
movie_ratings = ratings_df.merge(movies_df, on='movieId')

# Save the cleaned data
movie_ratings.to_csv('cleaned_movie_ratings.csv', index=False)
