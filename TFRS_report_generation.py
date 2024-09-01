import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

# Define the recommendation model
class RecommenderModel(tfrs.Model):
    def __init__(self, user_model, movie_model):
        super().__init__()
        self.movie_model = movie_model
        self.user_model = user_model
        self.embedding_dim = 32  # Set the embedding dimension

        # Define embedding variables manually
        self.user_embeddings = tf.Variable(
            initial_value=tf.random.uniform(
                shape=[user_model.vocabulary_size(), self.embedding_dim],
                minval=-1.0, maxval=1.0
            ),
            trainable=True
        )
        self.movie_embeddings = tf.Variable(
            initial_value=tf.random.uniform(
                shape=[movie_model.vocabulary_size(), self.embedding_dim],
                minval=-1.0, maxval=1.0
            ),
            trainable=True
        )
        
        # Define Dense layer manually
        self.ratings_weights = tf.Variable(
            initial_value=tf.random.uniform(
                shape=[self.embedding_dim, 1],
                minval=-1.0, maxval=1.0
            ),
            trainable=True
        )

    def compute_loss(self, features, training=False):
        # Unpack the features and labels
        features_dict, true_ratings = features

        # Access the userId and movieId from the dictionary
        user_ids = features_dict['userId']
        movie_ids = features_dict['movieId']

        # Get embeddings manually
        user_embeddings = tf.gather(self.user_embeddings, user_ids)
        movie_embeddings = tf.gather(self.movie_embeddings, movie_ids)

        # Compute dot product and predictions
        dot_product = tf.reduce_sum(user_embeddings * movie_embeddings, axis=1)
        predictions = tf.matmul(tf.expand_dims(dot_product, axis=1), self.ratings_weights)

        # Compute loss
        loss = tf.reduce_mean(tf.square(true_ratings - tf.squeeze(predictions)))
        return loss

# Function to load the dataset
def load_dataset():
    movie_ratings = pd.read_csv('TENSORFLOW/cleaned_movie_ratings.csv')
    return movie_ratings

# Function to generate recommendations
def generate_recommendations(model, user_ids, top_k=10):
    # Load movie IDs
    movie_ids = pd.read_csv('TENSORFLOW/cleaned_movie_ratings.csv')['movieId'].astype(str).unique()
    movie_index = tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=movie_ids,
            values=tf.range(len(movie_ids), dtype=tf.int64)
        ),
        num_oov_buckets=1
    )

    # Prepare datasets for user embeddings
    user_ids = tf.constant(user_ids)
    user_index = model.user_model

    # Generate recommendations
    user_embeddings = tf.gather(model.user_embeddings, user_ids)
    movie_embeddings = model.movie_embeddings

    # Compute scores
    scores = tf.matmul(user_embeddings, movie_embeddings, transpose_b=True)
    top_k_movie_indices = tf.argsort(scores, axis=-1, direction='DESCENDING')[:, :top_k]

    # Collect recommendations
    recommendations = []
    for user_id, movie_indices in zip(user_ids.numpy(), top_k_movie_indices):
        recommended_movies = [movie_ids[i] for i in movie_indices.numpy()]
        recommendations.append({
            'userId': user_id,
            'recommendedMovies': ', '.join(recommended_movies)
        })

    return recommendations

# Function to create and save the report
def create_report(model_path, output_csv_path):
    # Load the model
    model = tf.saved_model.load(model_path)

    # Load dataset and generate recommendations
    movie_ratings = load_dataset()
    user_ids = movie_ratings['userId'].astype(str).unique()
    recommendations = generate_recommendations(model, user_ids)

    # Convert recommendations to DataFrame
    recommendations_df = pd.DataFrame(recommendations)

    # Save the recommendations to CSV
    recommendations_df.to_csv(output_csv_path, index=False)
    print(f"Report saved to {output_csv_path}")

if __name__ == "__main__":
    model_path = 'TENSORFLOW/recommendation_model'
    output_csv_path = 'TENSORFLOW/recommendations_report.csv'
    create_report(model_path, output_csv_path)
