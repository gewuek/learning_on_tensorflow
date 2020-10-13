# Trace Tensor Class<br />

## Tensor Class<br />
***class Tensor(internal.NativeObject, core_tf_types.Tensor):<br />***
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/ops.py
```
  """A tensor is a multidimensional array of elements represented by a
  `tf.Tensor` object.  All elements are of a single known data type.
  When writing a TensorFlow program, the main object that is
  manipulated and passed around is the `tf.Tensor`.
  A `tf.Tensor` has the following properties:
  * a single data type (float32, int32, or string, for example)
  * a shape
  TensorFlow supports eager execution and graph execution.  In eager
  execution, operations are evaluated immediately.  In graph
  execution, a computational graph is constructed for later
  evaluation.
  TensorFlow defaults to eager execution.  In the example below, the
  matrix multiplication results are calculated immediately.
  >>> # Compute some values using a Tensor
  >>> c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
  >>> d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
  >>> e = tf.matmul(c, d)
  >>> print(e)
  tf.Tensor(
  [[1. 3.]
   [3. 7.]], shape=(2, 2), dtype=float32)
  Note that during eager execution, you may discover your `Tensors` are actually
  of type `EagerTensor`.  This is an internal detail, but it does give you
  access to a useful function, `numpy`:
  >>> type(e)
  <class '...ops.EagerTensor'>
  >>> print(e.numpy())
    [[1. 3.]
     [3. 7.]]
  In TensorFlow, `tf.function`s are a common way to define graph execution.
  A Tensor's shape (that is, the rank of the Tensor and the size of
  each dimension) may not always be fully known.  In `tf.function`
  definitions, the shape may only be partially known.
  Most operations produce tensors of fully-known shapes if the shapes of their
  inputs are also fully known, but in some cases it's only possible to find the
  shape of a tensor at execution time.
  A number of specialized tensors are available: see `tf.Variable`,
  `tf.constant`, `tf.placeholder`, `tf.sparse.SparseTensor`, and
  `tf.RaggedTensor`.
  For more on Tensors, see the [guide](https://tensorflow.org/guide/tensor).
  """
  ```
