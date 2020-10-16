# coding=utf-8
# Copyright 2020 The Uncertainty Metrics Authors.
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

# Lint as: python3
"""Tests for SemiParametric Calibration Error.

"""

from absl.testing import absltest
import numpy as np
import sklearn.model_selection
import uncertainty_metrics.numpy as um


class SemiparametricCalibrationErrorTest(absltest.TestCase):

  def test_zero_one(self):
    n = 2000
    probs = np.random.rand(n)
    calibration_error = 0.7 * probs ** 2 + 0.3 * probs
    # Simulate outcomes according to this model.
    labels = (np.random.rand(n) <= calibration_error).astype(np.integer)
    ce = um.SPCE(smoothing='spline')
    est = ce.binary_calibration_error(probs, labels)
    self.assertGreaterEqual(est, 0)
    self.assertLessEqual(est, 1)

  def test_simple_call(self):
    n = 2000
    probs = np.random.rand(n)
    calibration_error = 0.7 * probs ** 2 + 0.3 * probs
    # Simulate outcomes according to this model.
    labels = (np.random.rand(n) <= calibration_error).astype(np.integer)
    est = um.spce(probs, labels, smoothing='spline')
    self.assertGreaterEqual(est, 0)
    self.assertLessEqual(est, 1)

  def test_conf_int(self):
    n = 2000
    probs = np.random.rand(n)
    calibration_error = 0.7 * probs ** 2 + 0.3 * probs
    # Simulate outcomes according to this model.
    labels = (np.random.rand(n) <= calibration_error).astype(np.integer)
    lower_ci, _, upper_ci = um.spce_conf_int(
        probs, labels, smoothing='spline')
    self.assertGreaterEqual(lower_ci, 0)
    self.assertLessEqual(lower_ci, 1)
    self.assertGreaterEqual(upper_ci, 0)
    self.assertLessEqual(upper_ci, 1)

  def test_mean_plug_in(self):
    n = 2000
    probs = np.random.rand(n)
    calibration_error = 0.7 * probs ** 2 + 0.3 * probs
    # Continuous outcomes previously weren't allowed because StratifiedKFold
    # only allows discrete outcomes. Useful for testing to have an oracle
    # that passes in true calibration probabilities as outcomes, which are
    # continuous. Therefore, pass in a KFold object.
    ce = um.SPCE(smoothing='spline',
                 fold_generator=sklearn.model_selection.KFold(5, shuffle=True))
    est = ce.binary_calibration_error(probs, calibration_error)
    self.assertGreaterEqual(est, 0)
    self.assertLessEqual(est, 1)

  def test_shared_api(self):
    n = 2000
    probs = np.random.rand(n)
    calibration_error = 0.7 * probs ** 2 + 0.3 * probs
    # Simulate outcomes according to this model.
    labels = (np.random.rand(n) <= calibration_error).astype(np.integer)
    ce = um.SPCE(smoothing='spline')
    ce.update_state(labels, probs)
    est = ce.result()
    self.assertGreaterEqual(est, 0)
    self.assertLessEqual(est, 1)

  def test_multilabel(self):
    pred_probs = [
        [0.31, 0.32, 0.27],
        [0.37, 0.33, 0.30],
        [0.30, 0.31, 0.39],
        [0.61, 0.38, 0.01],
        [0.10, 0.65, 0.25],
        [0.91, 0.05, 0.04],
    ]
    # max_pred_probs: [0.32, 0.37, 0.39, 0.61, 0.65, 0.91]
    # pred_class: [1, 0, 2, 0, 1, 0]
    labels = [1., 0, 2., 1., 2., 0.]

    spce = um.SPCE(folds=2)
    spce.update_state([int(i) for i in labels], np.array(pred_probs))
    est = spce.result()
    self.assertGreaterEqual(est, 0)
    self.assertLessEqual(est, 1)


if __name__ == '__main__':
  absltest.main()
