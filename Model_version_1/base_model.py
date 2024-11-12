import tensorflow as tf

class BaseModel(tf.keras.Model):
    def __init__(self, output_dim=1, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)
        super(BaseModel, self).__init__()
        
        # Simple architecture for delta prediction
        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(16, activation='relu')
        self.output_layer = tf.keras.layers.Dense(output_dim)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.output_layer(x)