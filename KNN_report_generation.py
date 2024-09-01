import pandas as pd
import matplotlib.pyplot as plt

# Example: Load recommendation results and generate a summary
recommendations = pd.read_csv('recommendations.csv')

# Example: Generate a summary report
summary = recommendations.describe()
summary.to_csv('recommendation_summary.csv')

# Save a visualization
recommendations['rating'].hist()
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Recommendation Ratings Distribution')
plt.savefig('recommendation_ratings_distribution.png')
