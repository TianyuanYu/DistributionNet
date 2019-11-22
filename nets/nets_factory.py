# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

from nets import resnet_v1


slim = tf.contrib.slim

networks_map = {
                'resnet_v1_50': resnet_v1.resnet_v1_50,
                'resnet_v1_distributions_50': resnet_v1.resnet_v1_distributions_50,
                'resnet_v1_distributions_baseline_50': resnet_v1.resnet_v1_distributions_baseline_50,
               }

arg_scopes_map = {
                  'resnet_v1_50': resnet_v1.resnet_arg_scope,
                  'resnet_v1_distributions_50': resnet_v1.resnet_arg_scope,
                  'resnet_v1_distributions_baseline_50': resnet_v1.resnet_arg_scope,
                 }


def get_network_fn(name, num_classes, weight_decay=0.0, is_training=False, sample_number=1):
  """Returns a network_fn such as `logits, end_points = network_fn(images)`.

  Args:
    name: The name of the network.
    num_classes: The number of classes to use for classification.
    weight_decay: The l2 coefficient for the model weights.
    is_training: `True` if the model is being used for training and `False`
      otherwise.

  Returns:
    network_fn: A function that applies the model to a batch of images. It has
      the following signature:
        logits, end_points = network_fn(images)
  Raises:
    ValueError: If network `name` is not recognized.
  """
  if name not in networks_map:
    raise ValueError('Name of network unknown %s' % name)

  if name in arg_scopes_map:
      arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
  else:
      raise ValueError('Name of network unknown %s' % name)

  func = networks_map[name]
  @functools.wraps(func)
  def network_fn1(images):
    with slim.arg_scope(arg_scope):
      return func(images, num_classes, is_training=is_training, sample_number=sample_number)
  if hasattr(func, 'default_image_size'):
    network_fn1.default_image_size = func.default_image_size
  def network_fn2(images, refs):
    with slim.arg_scope(arg_scope):
      return func(images, refs, num_classes, is_training=is_training, sample_number=sample_number)
  if hasattr(func, 'default_image_size'):
    network_fn2.default_image_size = func.default_image_size
  if 'clean' in name:
      return network_fn2
  else:
      return network_fn1
