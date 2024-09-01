import pandas as pd

def preprocess_data():
    # Load data
    movie_ratings = pd.read_csv('TENSORFLOW/ratings.csv')
    movies = pd.read_csv('TENSORFLOW/movies.csv')
    tags = pd.read_csv('TENSORFLOW/tags.csv')

    # Merge datasets
    data = pd.merge(movie_ratings, movies, on='movieId')
    data = pd.merge(data, tags, on=['userId', 'movieId'], how='left')

    # Data cleaning
    data['tag'].fillna('', inplace=True)
    data['rating'] = data['rating'].astype(float)

    # Save cleaned data
    data.to_csv('TENSORFLOW/cleaned_movie_ratings.csv', index=False)

if __name__ == "__main__":
    preprocess_data()
