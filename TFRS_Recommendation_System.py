import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd

# Load and preprocess your dataset
def load_and_preprocess_data():
    # Load the dataset
    movie_ratings = pd.read_csv('TENSORFLOW/cleaned_movie_ratings.csv')
    
    # Extract unique user and movie IDs
    user_ids = movie_ratings['userId'].astype(str).unique()
    movie_ids = movie_ratings['movieId'].astype(str).unique()

    # Create TensorFlow datasets
    ratings = tf.data.Dataset.from_tensor_slices((
        {
            'userId': movie_ratings['userId'].astype(str).values,
            'movieId': movie_ratings['movieId'].astype(str).values
        },
        movie_ratings['rating'].values.astype(float)
    ))
    
    # Shuffle and batch the dataset
    ratings = ratings.shuffle(10000).batch(32)

    # Create lookup layers
    user_model = tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None)
    movie_model = tf.keras.layers.StringLookup(vocabulary=movie_ids, mask_token=None)

    return ratings, user_model, movie_model

# Define the recommendation model
class RecommenderModel(tfrs.Model):
    def __init__(self, user_model, movie_model):
        super().__init__()
        self.movie_model: tf.keras.layers.Layer = movie_model
        self.user_model: tf.keras.layers.Layer = user_model
        self.ratings: tf.keras.layers.Layer = tf.keras.layers.Dense(1, activation=None)
        self.embedding_dim = 32  # Set the embedding dimension

        # Define embedding layers
        self.user_embeddings = tf.keras.layers.Embedding(input_dim=user_model.vocabulary_size(), output_dim=self.embedding_dim)
        self.movie_embeddings = tf.keras.layers.Embedding(input_dim=movie_model.vocabulary_size(), output_dim=self.embedding_dim)

    def compute_loss(self, features, training=False):
        # Unpack the features and labels
        features_dict, true_ratings = features

        # Access the userId and movieId from the dictionary
        user_ids = features_dict['userId']
        movie_ids = features_dict['movieId']

        # Get embeddings
        user_embeddings = self.user_embeddings(user_ids)
        movie_embeddings = self.movie_embeddings(movie_ids)

        # Compute dot product and predictions
        dot_product = tf.reduce_sum(user_embeddings * movie_embeddings, axis=1)
        predictions = self.ratings(tf.expand_dims(dot_product, axis=1))

        # Compute loss
        loss = tf.keras.losses.MeanSquaredError()(true_ratings, tf.squeeze(predictions))
        return loss

def train_and_evaluate_model():
    ratings, user_model, movie_model = load_and_preprocess_data()

    model = RecommenderModel(user_model, movie_model)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

    # Adding a try-except block to catch and display any errors during training
    try:
        model.fit(ratings, epochs=5)
    except Exception as e:
        print("An error occurred during training:", e)
    
    # Save the model
    tf.saved_model.save(model, 'TENSORFLOW/recommendation_model')
    print("Model saved to TENSORFLOW/recommendation_model")

if __name__ == "__main__":
    train_and_evaluate_model()
