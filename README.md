# learning_on_tensorflow<br />
A record for learning TensorFlow and Keras<br />

### Trace the Con2D class<br />

Class ***Conv***<br />
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/convolutional.py<br />
```
class Conv(Layer):
```


Class ***Layer***<br />
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/base_layer.py
```
class Layer(module.Module, version_utils.LayerVersionSelector):
```
Exmaple of Custom Layer:<br />
```
  class SimpleDense(Layer):
    def __init__(self, units=32):
        super(SimpleDense, self).__init__()
        self.units = units
    def build(self, input_shape):  # Create the state of the layer (weights)
      w_init = tf.random_normal_initializer()
      self.w = tf.Variable(
          initial_value=w_init(shape=(input_shape[-1], self.units),
                               dtype='float32'),
          trainable=True)
      b_init = tf.zeros_initializer()
      self.b = tf.Variable(
          initial_value=b_init(shape=(self.units,), dtype='float32'),
          trainable=True)
    def call(self, inputs):  # Defines the computation from inputs to outputs
        return tf.matmul(inputs, self.w) + self.b
  # Instantiates the layer.
  linear_layer = SimpleDense(4)
  # This will also call `build(input_shape)` and create the weights.
  y = linear_layer(tf.ones((2, 2)))
  assert len(linear_layer.weights) == 2
  # These weights are trainable, so they're listed in `trainable_weights`:
  assert len(linear_layer.trainable_weights) == 2
```


Class ***Module***<br />
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/module/module.py
```
class Module(tracking.AutoTrackable):
```
Example of Custom layer:
```
  >>> class Dense(tf.Module):
  ...   def __init__(self, input_dim, output_size, name=None):
  ...     super(Dense, self).__init__(name=name)
  ...     self.w = tf.Variable(
  ...       tf.random.normal([input_dim, output_size]), name='w')
  ...     self.b = tf.Variable(tf.zeros([output_size]), name='b')
  ...   def __call__(self, x):
  ...     y = tf.matmul(x, self.w) + self.b
  ...     return tf.nn.relu(y)
  
  You can use the Dense layer as you would expect:
  >>> d = Dense(input_dim=3, output_size=2)
  >>> d(tf.ones([1, 3]))
  <tf.Tensor: shape=(1, 2), dtype=float32, numpy=..., dtype=float32)>

```


