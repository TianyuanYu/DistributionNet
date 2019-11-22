from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.ops import control_flow_ops

def _crop(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.

  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.

  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.

  Returns:
    the cropped (and resized) image.

  Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = tf.shape(image)

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3),
      ['Rank of image must be equal to 3.'])
  cropped_shape = control_flow_ops.with_dependencies(
      [rank_assertion],
      tf.stack([crop_height, crop_width, original_shape[2]]))

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  image = control_flow_ops.with_dependencies(
      [size_assertion],
      tf.slice(image, offsets, cropped_shape))
  return tf.reshape(image, cropped_shape)


def _random_crop(image_list, crop_height, crop_width):
  """Crops the given list of images.

  The function applies the same crop to each image in the list. This can be
  effectively applied when there are multiple image inputs of the same
  dimension such as:

    image, depths, normals = _random_crop([image, depths, normals], 120, 150)

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the new height.
    crop_width: the new width.

  Returns:
    the image_list with cropped images.

  Raises:
    ValueError: if there are multiple image inputs provided with different size
      or the images are smaller than the crop dimensions.
  """
  if not image_list:
    raise ValueError('Empty image_list.')

  # Compute the rank assertions.
  rank_assertions = []
  for i in range(len(image_list)):
    image_rank = tf.rank(image_list[i])
    rank_assert = tf.Assert(
        tf.equal(image_rank, 3),
        ['Wrong rank for tensor  %s [expected] [actual]',
         image_list[i].name, 3, image_rank])
    rank_assertions.append(rank_assert)

  image_shape = control_flow_ops.with_dependencies(
      [rank_assertions[0]],
      tf.shape(image_list[0]))
  image_height = image_shape[0]
  image_width = image_shape[1]
  crop_size_assert = tf.Assert(
      tf.logical_and(
          tf.greater_equal(image_height, crop_height),
          tf.greater_equal(image_width, crop_width)),
      ['Crop size greater than the image size.'])

  asserts = [rank_assertions[0], crop_size_assert]

  for i in range(1, len(image_list)):
    image = image_list[i]
    asserts.append(rank_assertions[i])
    shape = control_flow_ops.with_dependencies([rank_assertions[i]],
                                               tf.shape(image))
    height = shape[0]
    width = shape[1]

    height_assert = tf.Assert(
        tf.equal(height, image_height),
        ['Wrong height for tensor %s [expected][actual]',
         image.name, height, image_height])
    width_assert = tf.Assert(
        tf.equal(width, image_width),
        ['Wrong width for tensor %s [expected][actual]',
         image.name, width, image_width])
    asserts.extend([height_assert, width_assert])

  # Create a random bounding box.
  #
  # Use tf.random_uniform and not numpy.random.rand as doing the former would
  # generate random numbers at graph eval time, unlike the latter which
  # generates random numbers at graph definition time.
  max_offset_height = control_flow_ops.with_dependencies(
      asserts, tf.reshape(image_height - crop_height + 1, []))
  max_offset_width = control_flow_ops.with_dependencies(
      asserts, tf.reshape(image_width - crop_width + 1, []))
  offset_height = tf.random_uniform(
      [], maxval=max_offset_height, dtype=tf.int32)
  offset_width = tf.random_uniform(
      [], maxval=max_offset_width, dtype=tf.int32)

  return [_crop(image, offset_height, offset_width,
                crop_height, crop_width) for image in image_list]


def _upl_crop(image_list, crop_height, crop_width):
    # Upper Left
    outputs = []
    for image in image_list:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        offset_height = 0
        offset_width = 0

        outputs.append(_crop(image, offset_height, offset_width,
                             crop_height, crop_width))
    return outputs

def _upr_crop(image_list, crop_height, crop_width):
    # Upper Right
    outputs = []
    for image in image_list:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        offset_height = 0
        offset_width = image_width - crop_width

        outputs.append(_crop(image, offset_height, offset_width,
                             crop_height, crop_width))
    return outputs

def _dl_crop(image_list, crop_height, crop_width):
    # Down Left
    outputs = []
    for image in image_list:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        offset_height = image_height - crop_height
        offset_width = 0

        outputs.append(_crop(image, offset_height, offset_width,
                             crop_height, crop_width))
    return outputs

def _dr_crop(image_list, crop_height, crop_width):
    # Down Right
    outputs = []
    for image in image_list:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        offset_height = image_height - crop_height
        offset_width = image_width - crop_width

        outputs.append(_crop(image, offset_height, offset_width,
                             crop_height, crop_width))
    return outputs


def _central_crop(image_list, crop_height, crop_width):
  """Performs central crops of the given image list.

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    the list of cropped images.
  """
  outputs = []
  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2

    outputs.append(_crop(image, offset_height, offset_width,
                         crop_height, crop_width))
  return outputs


def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)


def _smallest_size_at_least(height, width, smallest_side):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.to_float(height)
  width = tf.to_float(width)
  smallest_side = tf.to_float(smallest_side)

  scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)
  return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=False)
  resized_image = tf.squeeze(resized_image)
  resized_image.set_shape([None, None, 3])
  return resized_image

def _resize(image, new_height, new_width):
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    if (height == new_height) and (width == new_width):
        image = tf.expand_dims(image, 0)
        resized_image = tf.squeeze(image)
        resized_image.set_shape([None, None, 3])
    else:
        image = tf.expand_dims(image, 0)
        resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                                 align_corners=False)
        resized_image = tf.squeeze(resized_image)
        resized_image.set_shape([None, None, 3])
    return resized_image


def augment_images_vgg(images):
    processed_images = tf.map_fn(lambda inputs: call_distort_image_vgg(inputs), elems=images, dtype=tf.float32)
    return processed_images


def call_distort_image_vgg(image):
    image = tf.to_float(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.subtract(image, [123.68, 116.78, 103.94])
    return image


def augment_images_inception(images, rand_erase=False):
    if rand_erase:
        processed_images = tf.map_fn(lambda inputs: call_distort_image_inception_with_erase(inputs), elems=images, dtype=tf.float32)
    else:
        processed_images = tf.map_fn(lambda inputs: call_distort_image_inception(inputs), elems=images, dtype=tf.float32)
    return processed_images


def call_distort_image_inception(image):
    image = tf.to_float(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.divide(image, 255)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def call_distort_image_inception_with_erase(image):
    image = tf.to_float(image)
    image = tf.image.random_flip_left_right(image)
    image = rand_erase_images(image)
    image = tf.divide(image, 255)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def call_distort_image(image):
    crop_size = FLAGS.crop_size
    if FLAGS.basenet in ['inceptionv1', 'inceptionv3']:
        image = tf.subtract(image, 0.5)
        image = tf.mul(image, 2.0)
    elif FLAGS.basenet in ['alexnet', 'resnet']:
        image = tf.subtract(image, [104.0069879317889, 116.66876761696767, 122.6789143406786])
    else:
        image = tf.subtract(image, 250.42)
    return distort_image(image, crop_size, crop_size)


def rand_erase_images(image, sl = 0.02, sh = 0.4, r1 = 0.3):

    image_height, image_width = int(image.shape[0]), int(image.shape[1])
    area = image_height * image_width
    target_area = tf.random_uniform([], sl, sh, name='area') * area
    aspect_ratio = tf.random_uniform([], r1, 1/r1, name='aspect_ratio')
    target_h = tf.minimum(tf.cast(tf.sqrt(target_area*aspect_ratio), tf.int32), image_height-2)
    target_w = tf.minimum(tf.cast(tf.sqrt(target_area/aspect_ratio), tf.int32), image_width-2)
    offset_x = tf.random_uniform([], 1, image_height-target_h, tf.int32, name='offset_x')
    offset_y = tf.random_uniform([], 1, image_width-target_w, tf.int32, name='offset_y')
    erase_cond = tf.less(tf.random_uniform([], 0, 1.0), .5)
    result_image = tf.cond(
        erase_cond,
        lambda: fix_erase_images(image, offset_x, offset_y, target_h, target_w),
        lambda: image)

    return result_image


def fix_erase_images(image, x, y, h, w):
    image_height, image_width, dim_size = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
    image_top = tf.slice(image, [0, 0, 0], [x, image_width, dim_size])
    image_left = tf.slice(image, [x, 0, 0], [h, y, dim_size])
    image_right = tf.slice(image, [x, y+w, 0], [h, image_width-y-w, dim_size])
    image_bottom = tf.slice(image, [x+h, 0, 0], [image_height-x-h, image_width, dim_size])
    random_image_part = tf.random_uniform([h, w, 3], 0, 1.0)
    image_between = tf.concat([image_left, random_image_part, image_right], 1)
    erased_image = tf.concat([image_top, image_between, image_bottom], 0)
    return tf.reshape(erased_image, [image_height, image_width, dim_size])


def process_images_vgg_for_eval(images):
    processed_images = tf.map_fn(lambda inputs: call_process_image_vgg_for_eval(inputs), elems=images, dtype=tf.float32)
    return processed_images


def call_process_image_vgg_for_eval(image):
    image = tf.to_float(image)
    image = tf.subtract(image, [123.68, 116.78, 103.94])
    return image


def process_images_inception_for_eval(images):
    processed_images = tf.map_fn(lambda inputs: call_process_image_inception_for_eval(inputs), elems=images, dtype=tf.float32)
    return processed_images


def call_process_image_inception_for_eval(image):
    image = tf.to_float(image)
    image = tf.divide(image, 255)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image