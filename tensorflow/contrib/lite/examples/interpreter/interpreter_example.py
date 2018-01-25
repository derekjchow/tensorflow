# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Example of using TFLite interpreter."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import misc
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper

FLAGS = app.flags.FLAGS
app.flags.DEFINE_string('model_path', '', 'Path to TFLite model file.')
app.flags.DEFINE_string('image_path', '', 'Path to image file.')
app.flags.DEFINE_string('output_path', '', 'Path to output file.')


def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)


def dump_tensor(file_name, output_data):
  with open(file_name, 'w+') as f:
    for value in  output_data.flatten():
      f.write(str(value) + '\n')


def main(_):
  interpreter = interpreter_wrapper.Interpreter(FLAGS.model_path)
  interpreter.allocate_tensors()

  image = rgb2gray(misc.imread(FLAGS.image_path))
  image = np.expand_dims(image, axis=0)
  image = np.expand_dims(image, axis=3)
  print(image.dtype)
  print(interpreter.get_input_details())
  interpreter.set_input('normalized_input_image_tensor', image)
  dump_tensor(FLAGS.output_path + '/normalized_input_image_tensor.tensor', image)
  interpreter.invoke()

  output_details = interpreter.get_output_details()
  for output_name in output_details.keys():
    output_data = interpreter.get_output(output_name)
    output_name = output_name.replace('/', '-')
    dump_tensor(FLAGS.output_path + '/' + output_name + '.tensor', output_data)


if __name__ == '__main__':
  app.run()
