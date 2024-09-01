import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    # Load the datasets
    movie_ratings = pd.read_csv('TENSORFLOW/cleaned_movie_ratings.csv')
    return movie_ratings

def plot_ratings_distribution(movie_ratings):
    plt.figure(figsize=(10, 6))
    sns.histplot(movie_ratings['rating'], bins=10, kde=True)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.savefig('TENSORFLOW/ratings_distribution.png')
    plt.show()

def plot_ratings_per_user(movie_ratings):
    user_ratings_count = movie_ratings.groupby('userId').size()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(user_ratings_count, bins=30, kde=True)
    plt.title('Number of Ratings per User')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Frequency')
    plt.savefig('TENSORFLOW/ratings_per_user.png')
    plt.show()

def plot_ratings_per_movie(movie_ratings):
    movie_ratings_count = movie_ratings.groupby('movieId').size()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(movie_ratings_count, bins=30, kde=True)
    plt.title('Number of Ratings per Movie')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Frequency')
    plt.savefig('TENSORFLOW/ratings_per_movie.png')
    plt.show()

def generate_summary_statistics(movie_ratings):
    summary = movie_ratings.describe(include='all')
    summary.to_csv('TENSORFLOW/ratings_summary.csv')

if __name__ == "__main__":
    movie_ratings = load_data()
    plot_ratings_distribution(movie_ratings)
    plot_ratings_per_user(movie_ratings)
    plot_ratings_per_movie(movie_ratings)
    generate_summary_statistics(movie_ratings)
