# learning_on_tensorflow<br />
A record for learning TensorFlow and Keras<br />

## Trace the Con2D class<br />

### Class Conv<br />
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/convolutional.py<br />
```
class Conv(Layer):
```


### Class Layer<br />
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


### Class Module<br />
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/module/module.py
```
class Module(tracking.AutoTrackable):
```
Example of Custom layer:
```
Base neural network module class.
  A module is a named container for `tf.Variable`s, other `tf.Module`s and
  functions which apply to user input. For example a dense layer in a neural
  network might be implemented as a `tf.Module`
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

### Class tracking.AutoTrackable<br />
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/tracking/tracking.py<br />
```
class AutoTrackable(base.Trackable):
```

### Class Trackable
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/tracking/base.py<br />
```
class Trackable(object):
```


## Trace Dense

### Class Dense
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/core.py
```
class Dense(Layer):
```
Call function:
```
  def call(self, inputs):
    return core_ops.dense(
        inputs,
        self.kernel,
        self.bias,
        self.activation,
        dtype=self._compute_dtype_object)
        
        ...
        
from tensorflow.python.keras.layers.ops import core as core_ops

```

### Class dense
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/ops/core.py
```
  rank = inputs.shape.rank
  if rank == 2 or rank is None:
    if isinstance(inputs, sparse_tensor.SparseTensor):
      outputs = sparse_ops.sparse_tensor_dense_matmul(inputs, kernel)
    else:
      outputs = gen_math_ops.mat_mul(inputs, kernel)
  # Broadcast kernel to inputs.
  else:
    outputs = standard_ops.tensordot(inputs, kernel, [[rank - 1], [0]])
    # Reshape the output back to the original ndim of the input.
    if not context.executing_eagerly():
      shape = inputs.shape.as_list()
      output_shape = shape[:-1] + [kernel.shape[-1]]
      outputs.set_shape(output_shape)
```

### Function sparse_tensor_dense_matmul
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/sparse_ops.py
```
def sparse_tensor_dense_matmul(sp_a,
                               b,
                               adjoint_a=False,
                               adjoint_b=False,
                               name=None):
  if isinstance(b, sparse_tensor.SparseTensor) \
          or isinstance(b, sparse_tensor.SparseTensorValue):
    # We can do C * D where C is sparse but if we want to do A * B when
    # B is sparse we have to transpose. But AB = (B'A')' so we have to feed in
    # the transpose of the arguments as well.
    if adjoint_a != adjoint_b:
      return array_ops.transpose(
          sparse_tensor_dense_matmul(b, sp_a, adjoint_a, adjoint_b))
    else:
      return array_ops.transpose(
          sparse_tensor_dense_matmul(
              b, sp_a, adjoint_a=not adjoint_a, adjoint_b=not adjoint_b))

  else:
    sp_a = _convert_to_sparse_tensor(sp_a)
    with ops.name_scope(name, "SparseTensorDenseMatMul",
                        [sp_a.indices, sp_a.values, b]) as name:
      b = ops.convert_to_tensor(b, name="b")
      return gen_sparse_ops.sparse_tensor_dense_mat_mul(
          a_indices=sp_a.indices,
          a_values=sp_a.values,
          a_shape=sp_a.dense_shape,
          b=b,
          adjoint_a=adjoint_a,
          adjoint_b=adjoint_b)
```
### gen_sparse_ops.py
gen_sparse_ops is generarted by bazel tools, described here:<br />
https://stackoverflow.com/questions/41147734/looking-for-source-code-of-from-gen-nn-ops-in-tensorflow<br />
The file can be on local installation:<br />
~/anaconda3/envs/tf1.5/lib/python3.6/site-packages/tensorflow_core/python/ops<br />
```
def sparse_tensor_dense_mat_mul(a_indices, a_values, a_shape, b, adjoint_a=False, adjoint_b=False, name=None):
  r"""Multiply SparseTensor (of rank 2) "A" by dense matrix "B".

  No validity checking is performed on the indices of A.  However, the following
  input format is recommended for optimal behavior:

  if adjoint_a == false:
    A should be sorted in lexicographically increasing order.  Use SparseReorder
    if you're not sure.
  if adjoint_a == true:
    A should be sorted in order of increasing dimension 1 (i.e., "column major"
    order instead of "row major" order).

  Args:
    a_indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2-D.  The `indices` of the `SparseTensor`, size `[nnz, 2]` Matrix.
    a_values: A `Tensor`.
      1-D.  The `values` of the `SparseTensor`, size `[nnz]` Vector.
    a_shape: A `Tensor` of type `int64`.
      1-D.  The `shape` of the `SparseTensor`, size `[2]` Vector.
    b: A `Tensor`. Must have the same type as `a_values`.
      2-D.  A dense Matrix.
    adjoint_a: An optional `bool`. Defaults to `False`.
      Use the adjoint of A in the matrix multiply.  If A is complex, this
      is transpose(conj(A)).  Otherwise it's transpose(A).
    adjoint_b: An optional `bool`. Defaults to `False`.
      Use the adjoint of B in the matrix multiply.  If B is complex, this
      is transpose(conj(B)).  Otherwise it's transpose(B).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a_values`.
  """
  _ctx = _context._context or _context.context()
  if _ctx is not None and _ctx._thread_local_data.is_eager:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._thread_local_data.device_name,
        "SparseTensorDenseMatMul", name, _ctx.post_execution_callbacks,
        a_indices, a_values, a_shape, b, "adjoint_a", adjoint_a, "adjoint_b",
        adjoint_b)
      return _result
    except _core._FallbackException:
      try:
        return sparse_tensor_dense_mat_mul_eager_fallback(
            a_indices, a_values, a_shape, b, adjoint_a=adjoint_a,
            adjoint_b=adjoint_b, name=name, ctx=_ctx)
      except _core._SymbolicException:
        pass  # Add nodes to the TensorFlow graph.
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)
  # Add nodes to the TensorFlow graph.
  if adjoint_a is None:
    adjoint_a = False
  adjoint_a = _execute.make_bool(adjoint_a, "adjoint_a")
  if adjoint_b is None:
    adjoint_b = False
  adjoint_b = _execute.make_bool(adjoint_b, "adjoint_b")
  _, _, _op = _op_def_lib._apply_op_helper(
        "SparseTensorDenseMatMul", a_indices=a_indices, a_values=a_values,
                                   a_shape=a_shape, b=b, adjoint_a=adjoint_a,
                                   adjoint_b=adjoint_b, name=name)
  _result = _op.outputs[:]
  _inputs_flat = _op.inputs
  _attrs = ("T", _op._get_attr_type("T"), "Tindices",
            _op._get_attr_type("Tindices"), "adjoint_a",
            _op.get_attr("adjoint_a"), "adjoint_b", _op.get_attr("adjoint_b"))
  _execute.record_gradient(
      "SparseTensorDenseMatMul", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result
```
Tools Trace:<br />
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/BUILD
```
tf_gen_op_wrapper_private_py(
    name = "sparse_ops_gen",
)

...

load("//tensorflow/python:build_defs.bzl", "tf_gen_op_wrapper_private_py")
```
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/build_defs.bzl
```
def tf_gen_op_wrapper_private_py(
        name,
        out = None,
        deps = [],
        require_shape_functions = False,
        visibility = []):
    if not name.endswith("_gen"):
        fail("name must end in _gen")
    if not visibility:
        visibility = ["//visibility:private"]
    bare_op_name = name[:-4]  # Strip off the _gen
    tf_gen_op_wrapper_py(
        name = bare_op_name,
        out = out,
        visibility = visibility,
        deps = deps,
        require_shape_functions = require_shape_functions,
        generated_target_name = name,
        api_def_srcs = [
            "//tensorflow/core/api_def:base_api_def",
            "//tensorflow/core/api_def:python_api_def",
        ],
    )
```
Function tf_gen_op_wrapper_py is defined here:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tensorflow.bzl
```
def tf_gen_op_wrapper_py(
        name,
        out = None,
        hidden = None,
        visibility = None,
        deps = [],
        require_shape_functions = False,
        hidden_file = None,
        generated_target_name = None,
        op_whitelist = [],
        cc_linkopts = lrt_if_needed(),
        api_def_srcs = [],
        compatible_with = []):
    _ = require_shape_functions  # Unused.

    if (hidden or hidden_file) and op_whitelist:
        fail("Cannot pass specify both hidden and op_whitelist.")

    # Construct a cc_binary containing the specified ops.
    tool_name = "gen_" + name + "_py_wrappers_cc"
    if not deps:
        deps = [str(Label("//tensorflow/core:" + name + "_op_lib"))]
    tf_cc_binary(
        name = tool_name,
        copts = tf_copts(),
        linkopts = if_not_windows(["-lm", "-Wl,-ldl"]) + cc_linkopts,
        linkstatic = 1,  # Faster to link this one-time-use binary dynamically
        visibility = [clean_dep("//tensorflow:internal")],
        deps = ([
            clean_dep("//tensorflow/core:framework"),
            clean_dep("//tensorflow/python:python_op_gen_main"),
        ] + deps),
    )

    # Invoke the previous cc_binary to generate a python file.
    if not out:
        out = "ops/gen_" + name + ".py"

    if hidden:
        op_list_arg = ",".join(hidden)
        op_list_is_whitelist = False
    elif op_whitelist:
        op_list_arg = ",".join(op_whitelist)
        op_list_is_whitelist = True
    else:
        op_list_arg = "''"
        op_list_is_whitelist = False

    # Prepare ApiDef directories to pass to the genrule.
    if not api_def_srcs:
        api_def_args_str = ","
    else:
        api_def_args = []
        for api_def_src in api_def_srcs:
            # Add directory of the first ApiDef source to args.
            # We are assuming all ApiDefs in a single api_def_src are in the
            # same directory.
            api_def_args.append(
                "$$(dirname $$(echo $(locations " + api_def_src +
                ") | cut -d\" \" -f1))",
            )
        api_def_args_str = ",".join(api_def_args)

    if hidden_file:
        # `hidden_file` is file containing a list of op names to be hidden in the
        # generated module.
        native.genrule(
            name = name + "_pygenrule",
            outs = [out],
            srcs = api_def_srcs + [hidden_file],
            exec_tools = [tool_name] + tf_binary_additional_srcs(),
            cmd = ("$(location " + tool_name + ") " + api_def_args_str +
                   " @$(location " + hidden_file + ") > $@"),
            compatible_with = compatible_with,
        )
    else:
        native.genrule(
            name = name + "_pygenrule",
            outs = [out],
            srcs = api_def_srcs,
            exec_tools = [tool_name] + tf_binary_additional_srcs(),
            cmd = ("$(location " + tool_name + ") " + api_def_args_str + " " +
                   op_list_arg + " " +
                   ("1" if op_list_is_whitelist else "0") + " > $@"),
            compatible_with = compatible_with,
        )

    # Make a py_library out of the generated python file.
    if not generated_target_name:
        generated_target_name = name
    native.py_library(
        name = generated_target_name,
        srcs = [out],
        srcs_version = "PY2AND3",
        visibility = visibility,
        deps = [
            clean_dep("//tensorflow/python:framework_for_generated_wrappers_v2"),
        ],
        # Instruct build_cleaner to try to avoid using this rule; typically ops
        # creators will provide their own tf_custom_op_py_library based target
        # that wraps this one.
        tags = ["avoid_dep"],
        compatible_with = compatible_with,
    )
```




