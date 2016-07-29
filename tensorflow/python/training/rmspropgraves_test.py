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

"""Tests for rmspropgraves."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import tensorflow as tf


class RMSPropGravesOptimizerTest(tf.test.TestCase):

  def testWithoutMomentum(self):
    for dtype in [tf.half, tf.float32]:
      with self.test_session():
        var0 = tf.Variable([1.0, 2.0], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)
        opt = tf.train.RMSPropGravesOptimizer(learning_rate=2.0, decay=0.9,
                                        momentum=0.0, epsilon=1.0)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        tf.initialize_all_variables().run()

        n0 = opt.get_slot(var0, "n_value")
        self.assertTrue(n0 is not None)
        n1 = opt.get_slot(var1, "n_value")
        self.assertTrue(n1 is not None)

        g0 = opt.get_slot(var0, "g_value")
        self.assertTrue(g0 is not None)
        g1 = opt.get_slot(var1, "g_value")
        self.assertTrue(g1 is not None)

        mom0 = opt.get_slot(var0, "momentum")
        self.assertTrue(mom0 is not None)
        mom1 = opt.get_slot(var1, "momentum")
        self.assertTrue(mom1 is not None)

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())
        # Step 1: the rms accumulators where 1. So we should see a normal
        # update: v -= grad * learning_rate
        update.run()

        # Check the n_value accumulators.
        self.assertAllCloseAccordingToType(np.array([0.901, 0.901]),
                                           n0.eval())
        self.assertAllCloseAccordingToType(np.array([0.90001, 0.90001]),
                                           n1.eval())

        # Check the g_value accumulators.
        self.assertAllCloseAccordingToType(np.array([0.91, 0.91]),
                                           g0.eval())
        self.assertAllCloseAccordingToType(np.array([0.901, 0.901]),
                                           g1.eval())

        # Check the parameters.
        self.assertAllCloseAccordingToType(
            np.array([1.0 - (0.1 * 2.0 / math.sqrt(0.901-0.91**2+1.0)),
                      2.0 - (0.1 * 2.0 / math.sqrt(0.901-0.91**2+1.0))]),
            var0.eval())
        self.assertAllCloseAccordingToType(
            np.array([3.0 - (0.01 * 2.0 / math.sqrt(0.90001-0.901**2+1.0)),
                      4.0 - (0.01 * 2.0 / math.sqrt(0.90001-0.901**2+1.0))]),
            var1.eval())

        # Step 2: the accumulators contain the previous update.
        update.run()
        # Check the n_value accumulators.
        self.assertAllCloseAccordingToType(
            np.array([0.901*0.9+0.001, 0.901*0.9+0.001]),
            n0.eval())
        self.assertAllCloseAccordingToType(
            np.array([0.90001*0.9+1e-5, 0.90001*0.9+1e-5]),
            n1.eval())

        # Check the g_value accumulators.
        self.assertAllCloseAccordingToType(
            np.array([0.91*0.9+0.01, 0.91*0.9+0.01]),
            g0.eval())
        self.assertAllCloseAccordingToType(
            np.array([0.901*0.9+1e-3, 0.901*0.9+1e-3]),
            g1.eval())

        # Check the parameters.
        self.assertAllCloseAccordingToType(
            np.array([1.0 - (0.1 * 2.0 / math.sqrt(0.901-0.91**2+1.0))
                      - (0.1 * 2.0 / math.sqrt((0.901*0.9+0.001)-(0.91*0.9+0.01)**2+1.0)),
                      2.0 - (0.1 * 2.0 / math.sqrt(0.901-0.91**2+1.0))
                      - (0.1 * 2.0 / math.sqrt((0.901*0.9+0.001)-(0.91*0.9+0.01)**2+1.0))]),
            var0.eval())
        self.assertAllCloseAccordingToType(
            np.array([3.0 - (0.01 * 2.0 / math.sqrt(0.90001-0.901**2+1.0))
                      - (0.01 * 2.0 / math.sqrt((0.90001*0.9+1e-5)-(0.901*0.9+1e-3)**2+1.0)),
                      4.0 - (0.01 * 2.0 / math.sqrt(0.90001-0.901**2+1.0))
                      - (0.01 * 2.0 / math.sqrt((0.90001*0.9+1e-5)-(0.901*0.9+1e-3)**2+1.0))]),
            var1.eval())

  def testWithMomentum(self):
    for dtype in [tf.half, tf.float32]:
      with self.test_session():
        var0 = tf.Variable([1.0, 2.0], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)

        opt = tf.train.RMSPropGravesOptimizer(learning_rate=2.0, decay=0.9,
                                        momentum=0.5, epsilon=1e-5)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        tf.initialize_all_variables().run()

        n0 = opt.get_slot(var0, "n_value")
        self.assertTrue(n0 is not None)
        n1 = opt.get_slot(var1, "n_value")
        self.assertTrue(n1 is not None)

        g0 = opt.get_slot(var0, "g_value")
        self.assertTrue(g0 is not None)
        g1 = opt.get_slot(var1, "g_value")
        self.assertTrue(g1 is not None)

        mom0 = opt.get_slot(var0, "momentum")
        self.assertTrue(mom0 is not None)
        mom1 = opt.get_slot(var1, "momentum")
        self.assertTrue(mom1 is not None)

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], var0.eval())
        self.assertAllClose([3.0, 4.0], var1.eval())
        # Step 1: rms = 1, mom = 0. So we should see a normal
        # update: v -= grad * learning_rate
        update.run()

        # Check the n_value accumulators.
        self.assertAllCloseAccordingToType(np.array([0.901, 0.901]),
                                           n0.eval())
        self.assertAllCloseAccordingToType(np.array([0.90001, 0.90001]),
                                           n1.eval())

        # Check the g_value accumulators.
        self.assertAllCloseAccordingToType(np.array([0.91, 0.91]),
                                           g0.eval())
        self.assertAllCloseAccordingToType(np.array([0.901, 0.901]),
                                           g1.eval())

        # Check the momentum accumulators
        self.assertAllCloseAccordingToType(
            np.array([(-0.1 * 2.0 / math.sqrt(0.901-0.91**2+1e-5)),
                      (-0.1 * 2.0 / math.sqrt(0.901-0.91**2+1e-5))]),
            mom0.eval())
        self.assertAllCloseAccordingToType(
            np.array([(-0.01 * 2.0/ math.sqrt(0.90001-0.901**2+1e-5)),
                      (-0.01 * 2.0/ math.sqrt(0.90001-0.901**2+1e-5))]),
            mom1.eval())

        # Check that the parameters.
        self.assertAllCloseAccordingToType(
            np.array([1.0 - (0.1 * 2.0 / math.sqrt(0.901-0.91**2+1e-5)),
                      2.0 - (0.1 * 2.0 / math.sqrt(0.901-0.91**2+1e-5))]),
            var0.eval())
        self.assertAllCloseAccordingToType(
            np.array([3.0 - (0.01 * 2.0/ math.sqrt(0.90001-0.901**2+1e-5)),
                      4.0 - (0.01 * 2.0/ math.sqrt(0.90001-0.901**2+1e-5))]),
            var1.eval())

        # Step 2: the root mean square accumulators contain the previous update.
        update.run()
        # Check the n_value accumulators.
        self.assertAllCloseAccordingToType(
            np.array([0.901*0.9+0.001, 0.901*0.9+0.001]),
            n0.eval())
        self.assertAllCloseAccordingToType(
            np.array([0.90001*0.9+1e-5, 0.90001*0.9+1e-5]),
            n1.eval())

        # Check the g_value accumulators.
        self.assertAllCloseAccordingToType(
            np.array([0.91*0.9+0.01, 0.91*0.9+0.01]),
            g0.eval())
        self.assertAllCloseAccordingToType(
            np.array([0.901*0.9+1e-3, 0.901*0.9+1e-3]),
            g1.eval())

        self.assertAllCloseAccordingToType(
            np.array([0.5 * (-0.1 * 2.0 / math.sqrt(0.901-0.91**2+1e-5)) -
                      (0.1*2.0/math.sqrt(0.901*0.9+0.001-(0.91*0.9+0.01)**2+1e-5)),
                      0.5 * (-0.1 * 2.0 / math.sqrt(0.901-0.91**2+1e-5)) -
                      (0.1*2.0/math.sqrt(0.901*0.9+0.001-(0.91*0.9+0.01)**2+1e-5))]),
            mom0.eval())
        self.assertAllCloseAccordingToType(
            np.array([0.5 * (-0.01 * 2.0 / math.sqrt(0.90001-0.901**2+1e-5)) -
                      (0.01 * 2.0 /math.sqrt(0.90001*0.9+1e-5-(0.901*0.9+1e-3)**2+1e-5)),
                      0.5 * (-0.01 * 2.0 / math.sqrt(0.90001-0.901**2+1e-5)) -
                      (0.01 * 2.0 / math.sqrt(0.90001*0.9+1e-5-(0.901*0.9+1e-3)**2+1e-5))]),
            mom1.eval())

        # Check the parameters.
        self.assertAllCloseAccordingToType(
            np.array([1.0 - (0.1 * 2.0 / math.sqrt(0.901-0.91**2+1e-5)) - (0.5 * (
                0.1 * 2.0 / math.sqrt(0.901-0.91**2+1e-5)) +(
                    0.1 * 2.0 / math.sqrt(0.901*0.9+0.001-(0.91*0.9+0.01)**2+1e-5))),
                      2.0 - (0.1 * 2.0 / math.sqrt(0.901-0.91**2+1e-5)) - (0.5 * (
                          0.1 * 2.0 / math.sqrt(0.901-0.91**2+1e-5)) +(
                              0.1 * 2.0 / math.sqrt(0.901*0.9+0.001-(0.91*0.9+0.01)**2+1e-5)))]),
            var0.eval())

        self.assertAllCloseAccordingToType(
            np.array([3.0 - (0.01 * 2.0 / math.sqrt(0.90001-0.901**2+1e-5))
                      - (0.5 *(0.01 * 2.0/ math.sqrt(0.90001-0.901**2+1e-5)) +
                         (0.01 * 2.0 /math.sqrt(0.90001*0.9+1e-5-(0.901*0.9+1e-3)**2+1e-5))),
                      4.0 - (0.01 * 2.0 / math.sqrt(0.90001-0.901**2+1e-5))
                      - (0.5 *(0.01 * 2.0/ math.sqrt(0.90001-0.901**2+1e-5)) +
                         (0.01 * 2.0 / math.sqrt(0.90001*0.9+1e-5-(0.901*0.9+1e-3)**2+1e-5)))]),
            var1.eval())


if __name__ == "__main__":
  tf.test.main()
