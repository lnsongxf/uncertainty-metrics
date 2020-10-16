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

"""Base calibration error.

Base calibration error class to be extended by both Semiparametric and General
calibration error.
"""

import abc
import numpy as np


def one_hot_encode(labels, num_classes=None):
  """One hot encoder for turning a vector of labels into a OHE matrix."""
  if num_classes is None:
    num_classes = len(np.unique(labels))
  return np.eye(num_classes)[labels]


def mean(inputs):
  """Be able to take the mean of an empty array without hitting NANs."""
  # pylint disable necessary for numpy and pandas
  if len(inputs) == 0:  # pylint: disable=g-explicit-length-test
    return 0
  else:
    return np.mean(inputs)


def binary_converter(probs):
  """Converts a binary probability vector into a matrix."""
  return np.array([[1-p, p] for p in probs])


class BaseCalibrationError(abc.ABC):
  """Base class for computing calibration error."""

  def __init__(self,
               max_prob,
               class_conditional,
               norm,
               threshold=0.0,
               distribution=None):
    self.max_prob = max_prob
    self.class_conditional = class_conditional
    self.norm = norm
    self.threshold = threshold
    self.distribution = distribution
    self.accuracies = None
    self.confidences = None
    self.calibration_error = None
    self.calibration_errors = None

  @abc.abstractmethod
  def binary_calibration_error(self, probs, binary_labels):
    pass

  def _format_labels_and_probs(self, labels, probs):
    """Format the labels and probs to have consistent matrix format."""
    probs = np.array(probs)
    labels = np.array(labels)
    if probs.ndim == 2:

      num_classes = probs.shape[1]
      if num_classes == 1:
        probs = probs[:, 0]
        probs = binary_converter(probs)
        num_classes = 2
    elif probs.ndim == 1:
      # Cover binary case
      probs = binary_converter(probs)
      num_classes = 2
    else:
      raise ValueError('Probs must have 1 or 2 dimensions.')

    # Convert the labels vector into a one-hot-encoded matrix.
    labels_matrix = one_hot_encode(labels, probs.shape[1])

    return labels_matrix, probs, num_classes

  def update_state(self, labels, probs):
    """Updates the value of the General Calibration Error."""
    labels_matrix, probs, num_classes = self._format_labels_and_probs(labels,
                                                                      probs)

    # When class_conditional is False, different classes are conflated.
    if not self.class_conditional:
      if self.max_prob:
        labels_matrix = labels_matrix[
            range(len(probs)), np.argmax(probs, axis=1)]
        probs = probs[range(len(probs)), np.argmax(probs, axis=1)]
      labels_matrix = labels_matrix[probs > self.threshold]
      probs = probs[probs > self.threshold]
      calibration_error = self.binary_calibration_error(probs.flatten(),
                                                        labels_matrix.flatten())

    # If class_conditional is true, predictions from different classes are
    # binned separately.
    else:
      # Initialize list for class calibration errors.
      class_calibration_error_list = []
      for j in range(num_classes):
        if not self.max_prob:
          probs_slice = probs[:, j]
          labels = labels_matrix[:, j]
          labels = labels[probs_slice > self.threshold]
          probs_slice = probs_slice[probs_slice > self.threshold]
          calibration_error = self.binary_calibration_error(
              probs_slice, labels)
          class_calibration_error_list.append(calibration_error/num_classes)
        else:
          # In the case where we use all datapoints,
          # max label has to be applied before class splitting.
          labels = labels_matrix[np.argmax(probs, axis=1) == j][:, j]
          probs_slice = probs[np.argmax(probs, axis=1) == j][:, j]
          labels = labels[probs_slice > self.threshold]
          probs_slice = probs_slice[probs_slice > self.threshold]
          calibration_error = self.binary_calibration_error(
              probs_slice, labels)
          class_calibration_error_list.append(calibration_error/num_classes)
      calibration_error = np.sum(class_calibration_error_list)

    if self.norm == 'l2':
      calibration_error = np.sqrt(calibration_error)

    self.calibration_error = calibration_error

  def result(self):
    return self.calibration_error

  def reset_state(self):
    self.calibration_error = None
