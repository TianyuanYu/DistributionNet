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
"""Contains definitions for the original form of Residual Networks.

The 'v1' residual networks (ResNets) implemented in this module were proposed
by:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Other variants were introduced in:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The networks defined in this module utilize the bottleneck building block of
[1] with projection shortcuts only for increasing depths. They employ batch
normalization *after* every weight layer. This is the architecture used by
MSRA in the Imagenet and MSCOCO 2016 competition models ResNet-101 and
ResNet-152. See [2; Fig. 1a] for a comparison between the current 'v1'
architecture and the alternative 'v2' architecture of [2] which uses batch
normalization *before* every weight layer in the so-called full pre-activation
units.

Typical use:

   from tensorflow.contrib.slim.nets import resnet_v1

ResNet-101 for image classification into 1000 classes:

   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(inputs, 1000, is_training=False)

ResNet-101 for semantic segmentation into 21 classes:

   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(inputs,
                                                21,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=16)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import resnet_utils
import tensorflow.contrib.slim as slim

resnet_arg_scope = resnet_utils.resnet_arg_scope
# slim = tf.contrib.slim

# v1: the main difference between v1 and v2 is bottleneck
@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
  """Bottleneck residual unit variant with BN after convolutions.

  This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
  its definition. Note that we use here the bottleneck variant which has an
  extra bottleneck layer.

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.

  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(inputs, depth, [1, 1], stride=stride,
                             activation_fn=None, scope='shortcut')

    residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                        rate=rate, scope='conv2')
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           activation_fn=None, scope='conv3')

    output = tf.nn.relu(shortcut + residual)

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.original_name_scope,
                                            output)


def resnet_v1(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              extra_fc_type=-1,
              extra_fc_out_dim=0,
              extra_fc_W_decay=0.0,
              f_decorr_fr=-1.,
              f_decorr_decay=0.0,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              avr_cc=0,
              feat_proj_type = -1,
              proj_dim = 1024,
              feat_prop_down=False,
              reuse=None,
              scope=None):
  """Generator for v1 ResNet models.

  This function generates a family of ResNet v1 models. See the resnet_v1_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce ResNets of various depths.

  Training for image classification on Imagenet is usually done with [224, 224]
  inputs, resulting in [7, 7] feature maps at the output of the last ResNet
  block for the ResNets defined in [1] that have nominal stride equal to 32.
  However, for dense prediction tasks we advise that one uses inputs with
  spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
  this case the feature maps at the ResNet output will have spatial shape
  [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
  and corners exactly aligned with the input image corners, which greatly
  facilitates alignment of the features to the image. Using as input [225, 225]
  images results in [8, 8] feature maps at the output of the last ResNet block.

  For dense prediction tasks, the ResNet needs to run in fully-convolutional
  (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
  have nominal stride equal to 32 and a good choice in FCN mode is to use
  output_stride=16 in order to increase the density of the computed features at
  small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a resnet_utils.Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks. If None
      we return the features before the logit layer.
    is_training: whether is training or not.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    include_root_block: If True, include the initial convolution followed by
      max-pooling, if False excludes it.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    avr_cc: 0, default, average; 1, concatenate

  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,
                         resnet_utils.stack_blocks_dense,
                         resnet_utils.extra_fc,
                         resnet_utils.projecting_feats],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([resnet_utils.extra_fc], loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES):
          with slim.arg_scope([slim.batch_norm], is_training=is_training):
            net = inputs
            if include_root_block:
              if output_stride is not None:
                if output_stride % 4 != 0:
                  raise ValueError('The output_stride needs to be a multiple of 4.')
                output_stride /= 4
              net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
              net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
            if extra_fc_type >= 0:
                # extra fc layer; keep its dimension as 4?
                net, pre_pool5 = resnet_utils.extra_fc(net, extra_fc_out_dim, extra_fc_W_decay, extra_fc_type, f_decorr_fr, f_decorr_decay)
            elif global_pool:
              # Global average pooling.
              net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
              deep_branch_feat = net
              # if (gate_proj_type != -1) and (gate_aug_type != -1):
              #     raise ValueError('Either gate_proj_type or gate_aug_type can be activated at a time.')
              #
              # if not gate_aug_type == -1:
              #     # Augmenting pool5 features with gates
              #     net = MoEL_utils.augmenting_gates(net, gate_aug_type, gate_prop_down, is_training,
              #                                       concat_gate_reg=concat_gate_reg,
              #                                       concat_gate_reg_type=concat_gate_reg_type)

              if not feat_proj_type == -1:
                  # projecting hidden feats and/or deep fc features to the same dimension and fuse them up
                  net = resnet_utils.projecting_feats(net, feat_proj_type, proj_dim, feat_prop_down, is_training,
                                                      avr_cc=avr_cc)

            if num_classes is not None:
              net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                normalizer_fn=None, scope='logits')
            if spatial_squeeze:
              logits = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
            # Convert end_points_collection into a dictionary of end_points.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            if num_classes is not None:
              end_points['predictions'] = slim.softmax(net, scope='predictions')
            if extra_fc_type >= 0:
                end_points['pre_pool5'] = pre_pool5
            elif global_pool:
                end_points['deep_branch_feat'] = deep_branch_feat
                end_points['PreLogits'] = tf.squeeze(deep_branch_feat, [1, 2], name='PreLogits')
            end_points['Logits'] = logits
            return logits, end_points
resnet_v1.default_image_size = 224


def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

def resnet_distributions_baseline_v1(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              output_stride=None,
              include_root_block=True,
              reuse=None,
              scope=None,
              sample_number = 1):
  """Generator for v1 ResNet models.

  This function generates a family of ResNet v1 models. See the resnet_v1_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce ResNets of various depths.

  Training for image classification on Imagenet is usually done with [224, 224]
  inputs, resulting in [7, 7] feature maps at the output of the last ResNet
  block for the ResNets defined in [1] that have nominal stride equal to 32.
  However, for dense prediction tasks we advise that one uses inputs with
  spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
  this case the feature maps at the ResNet output will have spatial shape
  [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
  and corners exactly aligned with the input image corners, which greatly
  facilitates alignment of the features to the image. Using as input [225, 225]
  images results in [8, 8] feature maps at the output of the last ResNet block.

  For dense prediction tasks, the ResNet needs to run in fully-convolutional
  (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
  have nominal stride equal to 32 and a good choice in FCN mode is to use
  output_stride=16 in order to increase the density of the computed features at
  small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a resnet_utils.Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks. If None
      we return the features before the logit layer.
    is_training: whether is training or not.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    include_root_block: If True, include the initial convolution followed by
      max-pooling, if False excludes it.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    avr_cc: 0, default, average; 1, concatenate

  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,
                         resnet_utils.stack_blocks_dense,
                         resnet_utils.extra_fc,
                         resnet_utils.projecting_feats],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([resnet_utils.extra_fc], loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES):
          with slim.arg_scope([slim.batch_norm], is_training=is_training):
            net = inputs
            if include_root_block:
              if output_stride is not None:
                if output_stride % 4 != 0:
                  raise ValueError('The output_stride needs to be a multiple of 4.')
                output_stride /= 4
              net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
              net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            with tf.variable_scope('Distributions'):
                mu = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                end_points['global_pool'] = mu

                sig = slim.conv2d(net, net.shape[-1], [net.shape[1], net.shape[2]],
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  biases_initializer=tf.zeros_initializer(),
                                  scope='sig',
                                  padding='VALID')

                mu = slim.dropout(mu, scope='Dropout', is_training=is_training)
                end_points['PreLogits_mean'] = tf.squeeze(mu, [1, 2], name='PreLogits_mean')
                end_points['PreLogits_sig'] = tf.squeeze(sig, [1, 2], name='PreLogits_sig')
                end_points['PreLogits'] = tf.squeeze(mu+sig, [1, 2], name='PreLogits')

                #tfd = tf.contrib.distributions
                #sample_dist = tfd.MultivariateNormalDiag(loc=end_points['PreLogits_mean'],
                #                                         scale_diag=end_points['PreLogits_sig'])

                #end_points['sample_dist'] = sample_dist

                if not num_classes:
                    return mu+sig, end_points

            logits = slim.conv2d(
                mu+sig,
                num_classes, [1, 1],
                activation_fn=None,
                normalizer_fn=None,
                biases_initializer=tf.zeros_initializer(),
                scope='logits')

            logits = tf.squeeze(logits, [1, 2])

            logits = tf.identity(logits, name='output')
            end_points['Logits'] = logits

            end_points['predictions'] = slim.softmax(logits, scope='predictions')
    return logits, end_points
resnet_distributions_baseline_v1.default_image_size = 224

def resnet_v1_distributions_baseline_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 sample_number=1,
                 scope='resnet_v1_50'):
  """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
  blocks = [
      resnet_utils.Block(
          'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      resnet_utils.Block(
          'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
      resnet_utils.Block(
          'block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
      resnet_utils.Block(
          'block4', bottleneck, [(2048, 512, 1)] * 3)
  ]
  return resnet_distributions_baseline_v1(inputs, blocks, num_classes, is_training, output_stride=output_stride,
                   include_root_block=True, reuse=reuse, scope=scope, sample_number=sample_number)
resnet_v1_distributions_baseline_50.default_image_size = resnet_v1.default_image_size


def resnet_distributions_v1(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              output_stride=None,
              include_root_block=True,
              reuse=None,
              scope=None,
              sample_number = 1):
  """Generator for v1 ResNet models.

  This function generates a family of ResNet v1 models. See the resnet_v1_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce ResNets of various depths.

  Training for image classification on Imagenet is usually done with [224, 224]
  inputs, resulting in [7, 7] feature maps at the output of the last ResNet
  block for the ResNets defined in [1] that have nominal stride equal to 32.
  However, for dense prediction tasks we advise that one uses inputs with
  spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
  this case the feature maps at the ResNet output will have spatial shape
  [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
  and corners exactly aligned with the input image corners, which greatly
  facilitates alignment of the features to the image. Using as input [225, 225]
  images results in [8, 8] feature maps at the output of the last ResNet block.

  For dense prediction tasks, the ResNet needs to run in fully-convolutional
  (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
  have nominal stride equal to 32 and a good choice in FCN mode is to use
  output_stride=16 in order to increase the density of the computed features at
  small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a resnet_utils.Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks. If None
      we return the features before the logit layer.
    is_training: whether is training or not.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    include_root_block: If True, include the initial convolution followed by
      max-pooling, if False excludes it.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    avr_cc: 0, default, average; 1, concatenate

  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,
                         resnet_utils.stack_blocks_dense,
                         resnet_utils.extra_fc,
                         resnet_utils.projecting_feats],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([resnet_utils.extra_fc], loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES):
          with slim.arg_scope([slim.batch_norm], is_training=is_training):
            net = inputs
            if include_root_block:
              if output_stride is not None:
                if output_stride % 4 != 0:
                  raise ValueError('The output_stride needs to be a multiple of 4.')
                output_stride /= 4
              net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
              net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            with tf.variable_scope('Distributions'):
                mu = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                end_points['global_pool'] = mu

                sig = slim.conv2d(net, net.shape[-1], [net.shape[1], net.shape[2]],
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  biases_initializer=tf.zeros_initializer(),
                                  scope='sig',
                                  padding='VALID')

                sig += 1e-10

                mu = slim.dropout(mu, scope='Dropout', is_training=is_training)
                end_points['PreLogits_mean'] = tf.squeeze(mu, [1, 2], name='PreLogits_mean')
                end_points['PreLogits_sig'] = tf.squeeze(sig, [1, 2], name='PreLogits_sig')

                tfd = tf.contrib.distributions
                #MultivariateNormalDiagWithSoftplusScale
                sample_dist = tfd.MultivariateNormalDiagWithSoftplusScale(loc=end_points['PreLogits_mean'],
                                                         scale_diag=end_points['PreLogits_sig'])

                end_points['sample_dist'] = sample_dist
                end_points['sample_dist_samples'] = sample_dist.sample(100)
                end_points['sample_dist_covariance'] = sample_dist.stddev()

                if not num_classes:
                    return mu, end_points

            logits = slim.conv2d(
                mu,
                num_classes, [1, 1],
                activation_fn=None,
                normalizer_fn=None,
                biases_initializer=tf.zeros_initializer(),
                scope='logits')

            logits = tf.squeeze(logits, [1, 2])

            #with tf.variable_scope('Distributions'):
            logits2 = []
            for iii in range(sample_number):
                z = sample_dist.sample(1)
                z = tf.reshape(z, [-1, int(mu.shape[-1])])

                    #import pdb
                    #pdb.set_trace()
                z=tf.expand_dims(z,1)
                z=tf.expand_dims(z,1)
                logits_tmp = slim.conv2d(
                    z,
                    num_classes, [1, 1],
                    activation_fn=None,
                    normalizer_fn=None,
                    biases_initializer=tf.zeros_initializer(),
                    scope='logits', 
                    reuse=True)
                logits2.append(tf.squeeze(logits_tmp, [1, 2])) 

            logits = tf.identity(logits, name='output')
            end_points['Logits'] = logits
            end_points['Logits2'] = logits2


            if sample_number == 1:
                end_points['predictions'] = slim.softmax(logits+0.1*logits2[0], scope='predictions')
            else:
                end_points['predictions'] = slim.softmax(logits, scope='predictions')

    return logits, logits2, end_points
resnet_distributions_v1.default_image_size = 224

def resnet_v1_distributions_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 sample_number=1,
                 scope='resnet_v1_50'):
  """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
  blocks = [
      resnet_utils.Block(
          'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      resnet_utils.Block(
          'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
      resnet_utils.Block(
          'block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
      resnet_utils.Block(
          'block4', bottleneck, [(2048, 512, 1)] * 3)
  ]
  return resnet_distributions_v1(inputs, blocks, num_classes, is_training, output_stride=output_stride,
                   include_root_block=True, reuse=reuse, scope=scope, sample_number=sample_number)
resnet_v1_distributions_50.default_image_size = resnet_v1.default_image_size


def resnet_v1_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v1_50',
                 sample_number=1):
  """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
  blocks = [
      resnet_utils.Block(
          'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      resnet_utils.Block(
          'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
      resnet_utils.Block(
          'block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
      resnet_utils.Block(
          'block4', bottleneck, [(2048, 512, 1)] * 3)
  ]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, reuse=reuse, scope=scope)
resnet_v1_50.default_image_size = resnet_v1.default_image_size
