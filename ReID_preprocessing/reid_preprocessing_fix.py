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
"""Provides utilities to preprocess images.

The preprocessing steps for VGG were introduced in the following technical
report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

import ReID_preprocessing.img_process_utils as img_proc_utils
import pdb

# pdb.set_trace()


# slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_RESIZE_H = 256
_RESIZE_W = 128


# _RESIZE_SIDE_MIN = 256
# _RESIZE_SIDE_MAX = 512


def preprocess_for_train(image,
                         output_height,
                         output_width):
    """Preprocesses the given image for training.

    Note that the actual resizing scale is sampled from
      [`resize_size_min`, `resize_size_max`].

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.

    Returns:
      A preprocessed image.
    """
    # image = img_proc_utils._resize(image, _RESIZE_H, _RESIZE_W)
    # image = img_proc_utils._random_crop([image], output_height, output_width)[0]
    # image.set_shape([output_height, output_width, 3])
    # image = tf.to_float(image)
    # image = tf.image.random_flip_left_right(image)
    # return img_proc_utils._mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])
    return img_proc_utils.augment_images_vgg(image)


def _post_img_list(crop_img_list):
    tmp_list = []
    for image in crop_img_list:
        # image.set_shape([output_height, output_width, 3])
        image = tf.to_float(image)
        # Remove Mean Here
        tmp_list.append(img_proc_utils._mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN]))
    return tmp_list


def preprocess_for_eval(image, output_height, output_width, test_mode):
    """Preprocesses the given image for evaluation.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      test_mode:
      1: crop central
      2: crop 4 corner + central and flip = 10 samples

    Returns:
      preprocessed image / images (testing) list!!!.
    """

    # remove the image resizing
    # image = tf.image.resize_images(image, [output_height, output_width])
    image = tf.image.resize_bilinear(image, [output_height, output_width])

    if test_mode == 1:

        # return a list
        return img_proc_utils.process_images_vgg_for_eval(image)
    elif test_mode == 2:
        # crop 4 corner + central and flip = 10 samples
        raise Exception('Not implemented')

        # return a list



def preprocess_image(image, output_height, output_width, test_mode, is_training=False):
    """Preprocesses the given image.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      is_training: `True` if we're preprocessing the image for training and
        `False` otherwise.

    Returns:
      preprocessed image / images (testing).
    """
    if is_training:
        return preprocess_for_train(image, output_height, output_width)
    else:
        return preprocess_for_eval(image, output_height, output_width, test_mode)
