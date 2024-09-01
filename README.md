# CineEssence: Movie Recommendation System

## Project Overview
CineEssence is a movie recommendation system designed to provide personalized movie suggestions to users based on their past ratings. The system utilizes a K-Nearest Neighbors (KNN) algorithm to predict user preferences and recommend movies they are likely to enjoy.

## Dataset
The project uses a dataset containing 100,836 ratings across 9,742 movies, provided by 610 users. The dataset includes four primary CSV files:

1. **movies.csv**: 
   - **Description**: Contains information about movies.
   - **Columns**: `movieId`, `title`, `genres`.
   - **Purpose**: Used to identify movies and their associated genres.

2. **ratings.csv**: 
   - **Description**: Contains user ratings for movies.
   - **Columns**: `userId`, `movieId`, `rating`, `timestamp`.
   - **Purpose**: Core data for training the recommendation model.

3. **tags.csv**: 
   - **Description**: Contains tags assigned by users to specific movies.
   - **Columns**: `userId`, `movieId`, `tag`, `timestamp`.
   - **Purpose**: Provides additional context and potential features for enhancing the recommendation system.

4. **links.csv**: 
   - **Description**: Provides external identifiers for movies.
   - **Columns**: `movieId`, `imdbId`, `tmdbId`.
   - **Purpose**: Allows for linking movie data with external databases like IMDb and TMDb.

## File Descriptions

### Data Processing
- **data_preprocessing**: 
   - **Description**: This file handles the preprocessing of the raw movie rating data. It cleans the data and prepares it for use in the recommendation algorithm.
   - **Output**: `cleaned_movie_ratings.csv`, which contains the preprocessed data ready for modeling.

### Exploratory Data Analysis (EDA)
- **eda**: 
   - **Description**: Performs exploratory data analysis on the cleaned data. It generates visualizations and insights, including rating distributions.
   - **Output**: Graphs and charts, such as the `rating_distribution` graph.

### Recommendation System
- **KNN_recommendation_system**: 
   - **Description**: Implements the K-Nearest Neighbors (KNN) algorithm to build the recommendation model. The model is trained using the cleaned movie ratings data to predict user preferences.
   - **Output**: `recommendation_model.pkl`, a serialized model file. It also generates `recommendations.csv`, which contains the predicted ratings for movies.

### Report Generation
- **KNN_report_generation**: 
   - **Description**: Generates a summary report based on the model's recommendations. This includes metrics like the count, mean, and standard deviation of the predicted ratings.
   - **Output**: `recommendation_summary.csv`, a summary of the recommendations.

## Algorithm
### K-Nearest Neighbors (KNN)
The KNN algorithm is a non-parametric, instance-based learning algorithm used for classification and regression. In this project, it is employed to predict movie ratings by finding the 'k' most similar users (neighbors) to a given user and aggregating their ratings for unseen movies. The algorithm works as follows:
1. **Training Phase**: The model learns the structure of the data by identifying patterns in the ratings provided by users.
2. **Prediction Phase**: For a given user, the model identifies the nearest neighbors and predicts ratings for movies the user hasn't rated yet, based on the neighbors' ratings.

### Outputs
- **recommendations.csv**: Contains the predicted ratings for each movie based on the KNN model.
- **recommendation_summary.csv**: Provides a statistical summary of the recommendations, including count, mean, and standard deviation.

## How to Use
To utilize this project, follow these steps:
1. **Preprocess the Data**: Run the data preprocessing script to generate the cleaned dataset.
2. **Perform EDA**: Use the EDA script to explore the data and understand the distribution of ratings.
3. **Train the Model**: Execute the recommendation system script to build and save the model.
4. **Generate Reports**: Run the report generation script to create summaries and gain insights from the recommendations.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more information.
