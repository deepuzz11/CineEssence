import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned data
movie_ratings = pd.read_csv('cleaned_movie_ratings.csv')

# Example analysis: Distribution of ratings
plt.hist(movie_ratings['rating'], bins=10)
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Ratings')
plt.savefig('rating_distribution.png')
plt.show()
