# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""One-line documentation for rmspropgraves module.

A.Graves's rmsprop algorithm [http://arxiv.org/pdf/1308.0850v5.pdf]

A detailed description of rmspropgraves.

- maintain a moving (discounted) average of the square of gradients
- maintain a moving (discounted) average of the gradients
- divide gradient by the root of the difference between these two averages

n = decay * n{t-1} + (1-decay) * gradient ** 2
g = decay * g{t-1} + (1-decay) * gradient
d = momentum * d{t-1} - learning_rate * g_t / sqrt(n - g ** 2 + epsilon)
delta = d

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class RMSPropGravesOptimizer(optimizer.Optimizer):
  """Optimizer that implements the Alex Graves's RMSProp algorithm.

  See the [paper]
  (http://arxiv.org/pdf/1308.0850v5.pdf).

  @@__init__
  """

  def __init__(self,
               learning_rate,
               decay=0.95,
               momentum=0.9,
               epsilon=1e-4,
               use_locking=False,
               name="RMSPropGraves"):
    """Construct a new RMSPropGraves optimizer.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      decay: Discounting factor for the history/coming gradient
      momentum: A scalar tensor.
      epsilon: Small value to avoid zero denominator.
      use_locking: If True use locks for update operation.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "RMSPropGraves".
    """
    super(RMSPropGravesOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._decay = decay
    self._momentum = momentum
    self._epsilon = epsilon

    # Tensors for learning rate and momentum.  Created in _prepare.
    self._learning_rate_tensor = None
    self._decay_tensor = None
    self._momentum_tensor = None
    self._epsilon_tensor = None

  def _create_slots(self, var_list):
    for v in var_list:
      val = constant_op.constant(1.0, dtype=v.dtype, shape=v.get_shape())
      self._get_or_make_slot(v, val, "n_value", self._name)
      self._get_or_make_slot(v, val, "g_value", self._name)
      self._zeros_slot(v, "momentum", self._name)

  def _prepare(self):
    self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                       name="learning_rate")
    self._decay_tensor = ops.convert_to_tensor(self._decay, name="decay")
    self._momentum_tensor = ops.convert_to_tensor(self._momentum,
                                                  name="momentum")
    self._epsilon_tensor = ops.convert_to_tensor(self._epsilon,
                                                 name="epsilon")

  def _apply_dense(self, grad, var):
    n = self.get_slot(var, "n_value")
    g = self.get_slot(var, "g_value")
    mom = self.get_slot(var, "momentum")
    return training_ops.apply_rms_prop_graves(
        var, n, g, mom,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        math_ops.cast(self._decay_tensor, var.dtype.base_dtype),
        math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
        math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
        grad, use_locking=self._use_locking).op

  def _apply_sparse(self, grad, var):
    raise NotImplementedError()
