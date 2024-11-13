import tensorflow as tf

class BaseModel(tf.keras.Model):
    def __init__(self, learning_rate=0.0001):
        super(BaseModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(64, activation='relu')
        self.dense4 = tf.keras.layers.Dense(32, activation='relu')
        self.dense5 = tf.keras.layers.Dense(16, activation='relu')
        self.output_layer = tf.keras.layers.Dense(9)
        self.learning_rate = learning_rate
        
    def get_optimizer(self):
        return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        return self.output_layer(x)