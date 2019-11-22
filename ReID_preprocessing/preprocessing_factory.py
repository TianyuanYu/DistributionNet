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

from ReID_preprocessing import reid_preprocessing_fix

def get_preprocessing(name, aug_mode=1, test_mode=1, is_training=False, rand_erase=False):
    """Returns preprocessing_fn(image, height, width, **kwargs).

      Args:
        name: The name of the preprocessing function.
        is_training: `True` if the model is being used for training and `False`
          otherwise.

      Returns:
        preprocessing_fn: A function that preprocessing a single image (pre-batch).
          It has the following signature:
            image = preprocessing_fn(image, output_height, output_width, ...).

      Raises:
        ValueError: If Preprocessing `name` is not recognized.
      """

    assert test_mode in [1, 2]

    if aug_mode == 3:
        # directly resize to fix shape
        # train: resize to 256 x 128, then random flip
        # testing: resize to 256 x 128
        # test_mode 1: just resize
        # test_mode 2: flip = 2 samples
        preprocessing_fn_map = {
            'resnet_v1_50': reid_preprocessing_fix,
            'resnet_v1_distributions_50': reid_preprocessing_fix,
            'resnet_v1_distributions_baseline_50': reid_preprocessing_fix,
        }
    else:
        raise ValueError('aug_mode should be set to 3.')

    if name not in preprocessing_fn_map:
        raise ValueError('Preprocessing name [%s] was not recognized' % name)

    def preprocessing_fn(image, output_height, output_width, **kwargs):
        if rand_erase:
            return preprocessing_fn_map[name].preprocess_image(
                image, output_height, output_width, test_mode, is_training=is_training, rand_erase=True, **kwargs)
        else:
            return preprocessing_fn_map[name].preprocess_image(
                image, output_height, output_width, test_mode, is_training=is_training, **kwargs)

    return preprocessing_fn
