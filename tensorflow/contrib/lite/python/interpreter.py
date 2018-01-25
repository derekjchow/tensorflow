# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Python TF-Lite interpreter."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.lite.python.interpreter_wrapper import tensorflow_wrap_interpreter_wrapper as interpreter_wrapper


def _get_numpy_type(i):
  if i == 1:
    return np.float32
  if i == 2:
    return np.int32
  if i == 3:
    return np.uint8
  if i == 4:
    return np.int64
  raise ValueError('Could not map in to numpy type: {}'.format(i))


class Interpreter(object):
  """Interpreter inferace for TF-Lite Models."""

  def __init__(self, model_path):
    """Constructor.

    Args:
      model_path: Path to TF-Lite Flatbuffer file.

    Raises:
      ValueError: If the interpreter was unable to open the model.
    """
    self._interpreter = interpreter_wrapper.CreateWrapperCPP(model_path)
    if not self._interpreter:
      raise ValueError('Failed to open {}'.format(model_path))

  def allocate_tensors(self):
    if not self._interpreter.AllocateTensors():
      raise ValueError('Failed to allocate tensors')

  def get_input_details(self):
    input_details = {}
    for i in range(self._interpreter.NumInputs()):
      input_name = self._interpreter.InputName(i)
      input_size = self._interpreter.InputSize(i)
      input_type = self._interpreter.InputType(i)
      input_details[input_name] = {
        'index': i,
        'shape': input_size,
        'dtype': _get_numpy_type(input_type),
      }
    return input_details

  def set_input(self, input_name, value):
    """Sets the value of the input.

    Args:
      input_name: Name of input tensor to set.
      value: Value of input tensor to set.

    Raises:
      ValueError: If the interpreter was set the input tensor.
    """
    input_details = self.get_input_details()
    if input_name not in input_details.keys():
      raise ValueError('{} not listed in model inputs'.format(input_name))
    input_shape = input_details[input_name]['shape']
    if np.array(value.shape != input_shape).any():
      raise ValueError('Unable to set {}: shape mismatch'.format(input_name))

    input_index = input_details[input_name]['index']
    if value.dtype == np.uint8:
      self._interpreter.SetInputUint8(input_index, value.flatten())
    elif value.dtype == np.float32:
      self._interpreter.SetInputFloat32(input_index, value.flatten())
    else:
      raise ValueError('Unsupported dtype for inference')


  def get_output_details(self):
    output_details = {}
    for i in range(self._interpreter.NumOutputs()):
      output_name = self._interpreter.OutputName(i)
      output_size = self._interpreter.OutputSize(i)
      output_type = self._interpreter.OutputType(i)
      output_details[output_name] = {
        'index': i,
        'shape': output_size,
        'dtype': _get_numpy_type(output_type),
      }
    return output_details

  def get_output(self, output_name):
    """Sets the value of the input.

    Args:
      output_name: Name of input tensor to get.

    Returns:
      a numpy array.

    Raises:
      ValueError: If there is no output name output_name.
    """
    output_details = self.get_output_details()
    if output_name not in output_details.keys():
      raise ValueError('{} not listed in model outputs'.format(output_name))

    output_shape = output_details[output_name]['shape']
    output_index = output_details[output_name]['index']
    output_type = output_details[output_name]['dtype']

    if output_type == np.uint8:
      output_data = self._interpreter.GetOutputUint8(output_index)
    elif output_type == np.float32:
      output_data = self._interpreter.GetOutputFloat32(output_index)
    else:
      raise ValueError('Unsupported output dtype')
    return np.reshape(output_data, output_shape)

  def invoke(self):
    if not self._interpreter.Invoke():
      raise ValueError('Failed to invoke TFLite model')
