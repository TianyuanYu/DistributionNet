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
"""Contains building blocks for various versions of Residual Networks.

Residual networks (ResNets) were proposed in:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Deep Residual Learning for Image Recognition. arXiv:1512.03385, 2015

More variants were introduced in:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Identity Mappings in Deep Residual Networks. arXiv: 1603.05027, 2016

We can obtain different ResNet variants by changing the network depth, width,
and form of residual unit. This module implements the infrastructure for
building them. Concrete ResNet units and full ResNet networks are implemented in
the accompanying resnet_v1.py and resnet_v2.py modules.

Compared to https://github.com/KaimingHe/deep-residual-networks, in the current
implementation we subsample the output activations in the last residual unit of
each block, instead of subsampling the input activations in the first residual
unit of each block. The two implementations give identical results but our
implementation is more memory efficient.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import collections
import tensorflow as tf
# import tensorflow.contrib.slim as slim
slim = tf.contrib.slim

import pdb


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """A named tuple describing a ResNet block.

  Its parts are:
    scope: The scope of the `Block`.
    unit_fn: The ResNet unit function which takes as input a `Tensor` and
      returns another `Tensor` with the output of the ResNet unit.
    args: A list of length equal to the number of units in the `Block`. The list
      contains one (depth, depth_bottleneck, stride) tuple for each unit in the
      block to serve as argument to unit_fn.
  """


def subsample(inputs, factor, scope=None):
  """Subsamples the input along the spatial dimensions.

  Args:
    inputs: A `Tensor` of size [batch, height_in, width_in, channels].
    factor: The subsampling factor.
    scope: Optional variable_scope.

  Returns:
    output: A `Tensor` of size [batch, height_out, width_out, channels] with the
      input, either intact (if factor == 1) or subsampled (if factor > 1).
  """
  if factor == 1:
    return inputs
  else:
    return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
  """Strided 2-D convolution with 'SAME' padding.

  When stride > 1, then we do explicit zero-padding, followed by conv2d with
  'VALID' padding.

  Note that

     net = conv2d_same(inputs, num_outputs, 3, stride=stride)

  is equivalent to

     net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
     net = subsample(net, factor=stride)

  whereas

     net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')

  is different when the input's height or width is even, which is why we add the
  current function. For more details, see ResnetUtilsTest.testConv2DSameEven().

  Args:
    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
    num_outputs: An integer, the number of output filters.
    kernel_size: An int with the kernel_size of the filters.
    stride: An integer, the output stride.
    rate: An integer, rate for atrous convolution.
    scope: Scope.

  Returns:
    output: A 4-D tensor of size [batch, height_out, width_out, channels] with
      the convolution output.
  """
  if stride == 1:
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,
                       padding='SAME', scope=scope)
  else:
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(inputs,
                    [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                       rate=rate, padding='VALID', scope=scope)


@slim.add_arg_scope
def extra_fc(in_net, extra_fc_out_dim, extra_fc_W_decay, extra_fc_type, f_decorr_fr, f_decorr_decay, outputs_collections=None,
                  loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES):
    """
    input 4 dimension, output 4 dimension
    fc using 2 dimension, weights only no bias
    :param in_net: 4 dim? squeeze
    :param extra_fc_out_dim: >0
    :param extra_fc_W_decay: >0
    :return:
        4 dim, extending it at last
    """
    assert extra_fc_out_dim > 0
    assert extra_fc_W_decay > 0
    assert extra_fc_type >= 0
    assert len(in_net.get_shape().as_list()) == 4

    # Global average pooling as input; dimension is kept, input as pre_pool5
    net = tf.reduce_mean(in_net, [1, 2], name='pre_pool5', keep_dims=True)
    pre_pool5 = net
    net = tf.squeeze(net, [1, 2])
    assert len(net.get_shape().as_list()) == 2

    # extra fc layer
    net = fc_layer(net, extra_fc_out_dim, extra_fc_W_decay, extra_fc_type, f_decorr_fr, f_decorr_decay,
                   outputs_collections, loss_collection, name='final_extra_fc')
    assert len(net.get_shape().as_list()) == 2

    # output as pool5
    net = tf.expand_dims(tf.expand_dims(net, axis=1), axis=1, name='pool5')
    assert len(net.get_shape().as_list()) == 4
    return net, pre_pool5


def fc_layer(in_net, extra_fc_out_dim, extra_fc_W_decay, extra_fc_type, f_decorr_fr, f_decorr_decay,
             outputs_collections, loss_collection, name='extra_fc'):
    """
    no bias
    :param in_net: 2 dim
    :param extra_fc_out_dim:
    :param extra_fc_W_decay:
    :param extra_fc_type:
    0. normal extra fc layer
    1. soft whitening reg ||W'W-I||_{F}^{2}
    2. soft W decorrelation (L1 loss)
    3. std W then soft decorrelation (L1 loss)
    :param outputs_collections:
    :param loss_collection:
    :param name:
    :return:
    """
    in_dim = in_net.get_shape().as_list()[-1]
    with tf.variable_scope(name) as sc:
        if extra_fc_type == 0:
            # normal extra fc layer
            net = slim.fully_connected(in_net,
                                 extra_fc_out_dim,
                                 activation_fn=tf.nn.relu,
                                 normalizer_fn=slim.batch_norm,
                                 weights_regularizer = slim.l2_regularizer(extra_fc_W_decay),
                                 weights_initializer = slim.xavier_initializer(),
                                 biases_initializer=None)
        else:
            # no need weights_regularizer
            net = slim.fully_connected(in_net,
                                       extra_fc_out_dim,
                                       activation_fn=tf.nn.relu,
                                       normalizer_fn=slim.batch_norm,
                                       weights_regularizer=None,
                                       weights_initializer=slim.xavier_initializer(),
                                       biases_initializer=None)

            # scope = sc.original_name_scope
            # another regularisation decay needed??? for feature reg
            net_decorr_reg(net, f_decorr_fr, f_decorr_decay, loss_collection)
            weights_list = get_weights(sc.original_name_scope)
            assert len(weights_list) == 1
            biases_list = get_biases(sc.original_name_scope)
            assert len(biases_list) == 0
            weight_decorr_whiten_reg(extra_fc_type, weights_list, in_dim, extra_fc_out_dim, extra_fc_W_decay, loss_collection)


    # checking
    weights_list = get_weights(sc.original_name_scope)
    assert len(weights_list) == 1
    biases_list = get_biases(sc.original_name_scope)
    assert len(biases_list) == 0
    # pdb.set_trace()
    assert net.get_shape().as_list()[-1] == extra_fc_out_dim
    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.original_name_scope,
                                            net)

def net_decorr_reg(in_net, f_decorr_fr, f_decorr_decay, loss_collection):
    # f_decorr_fr: for the accumulated covariance term.
    # in_net: dim 2
    # averaging the sample num (not yet)
    assert len(in_net.get_shape().as_list()) == 2
    in_dim = in_net.get_shape().as_list()[-1]

    if f_decorr_fr < 0.:
        print('no need feat decorrelation.')
    elif f_decorr_fr > 1:
        raise ValueError('f_decorr_fr should not > 1.')
    else: # 0=<f_decorr_fr<=1
        assert f_decorr_decay > 0.
        accum_cov_mat = tf.Variable(initial_value=tf.zeros([in_dim, in_dim]), name="accum_cov_mat")
        accum_aver_num = tf.Variable(initial_value=0.0, name="accum_aver_num")
        # averaged
        tmp_loss = moving_average_cov_mat(in_net, accum_cov_mat, accum_aver_num, f_decorr_fr)
        loss = tf.multiply(tf.constant(f_decorr_decay, dtype=tf.float32), tmp_loss)
        tf.add_to_collection(loss_collection, loss)

def moving_average_cov_mat(in_net, accum_cov_mat, accum_aver_num, f_decorr_fr):
    # moving averaging feature covariance mat
    # std the input first
    assert len(in_net.get_shape().as_list()) == 2
    in_dim = in_net.get_shape().as_list()[-1]
    N = in_net.get_shape().as_list()[0]
    assert N > 1
    mean, var = tf.nn.moments(in_net, [0], keep_dims=True)
    assert len(mean.get_shape().as_list()) == len(var.get_shape().as_list()) == 2
    assert mean.get_shape().as_list()[1] == var.get_shape().as_list()[1] == in_dim
    in_std_tensor = tf.div(tf.subtract(in_net, mean), tf.add(tf.sqrt(var), tf.constant(0.000001, dtype=tf.float32)))

    # accumulate cov mat
    fr_tensor = tf.constant(f_decorr_fr, dtype=tf.float32)
    #averaging the sample num
    cov_mat = tf.div( tf.matmul(in_std_tensor, in_std_tensor, transpose_a=True), tf.constant(N-1, dtype=tf.float32) )
    tmp_accum_cov_mat = tf.add(tf.multiply(fr_tensor, accum_cov_mat), cov_mat)
    # accumulate num
    tmp_accum_aver_num = tf.add(tf.multiply(fr_tensor, accum_aver_num), tf.constant(1.0, dtype=tf.float32))
    # averageing
    # aver_accum_cov_mat = tf.div(accum_cov_mat, accum_aver_num)
    # absolute for loss computation
    abs_aver_accum_cov_mat = tf.abs(tf.div(tmp_accum_cov_mat, tmp_accum_aver_num))
    assert len(abs_aver_accum_cov_mat.get_shape().as_list()) == 2
    assert abs_aver_accum_cov_mat.get_shape().as_list()[0] == abs_aver_accum_cov_mat.get_shape().as_list()[1] == in_dim
    # update through assignment
    accum_cov_mat.assign(tmp_accum_cov_mat)
    accum_aver_num.assign(tmp_accum_aver_num)


    tmp_loss = tf.subtract(tf.reduce_sum(abs_aver_accum_cov_mat, [0, 1]), tf.trace(abs_aver_accum_cov_mat))
    # averaging the dim num: in_dim*(in_dim-1)
    loss = tf.div(tmp_loss, tf.constant(in_dim*(in_dim-1), dtype=tf.float32))
    return loss


def get_weights(name):
    # pdb.set_trace()
    return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name) if v.name.endswith('weights:0')]

def get_biases(name):
    # pdb.set_trace()
    return [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name) if v.name.endswith('bias:0')]

def weight_decorr_whiten_reg(extra_fc_type, weights_list, in_dim, out_dim, extra_fc_W_decay, loss_collection):
    """
    :param extra_fc_type:
    1. soft whitening reg ||W'W-I||_{F}^{2}
    2. soft W decorrelation (L1 loss)
    3. std W then soft decorrelation (L1 loss)
    :param weights_list:
    :param in_dim:
    :param extra_fc_out_dim:
    :param extra_fc_W_decay:
    :param loss_collection:
    :return: None
    """

    #get the weight and check size
    W_tensor = weights_list[0]
    # pdb.set_trace()
    assert len(W_tensor.get_shape().as_list()) == 2
    assert W_tensor.get_shape().as_list()[0] == in_dim
    assert W_tensor.get_shape().as_list()[1] == out_dim
    assert in_dim >= out_dim
    assert in_dim > 1

    if extra_fc_type == 1:
        # soft whitening reg ||W'W-I||_{F}^{2}
        I_tensor = tf.eye(out_dim)
        # averging sample num
        Cov_tensor = tf.div(tf.matmul(W_tensor, W_tensor, transpose_a=True), tf.constant(in_dim-1, dtype=tf.float32))
        diff_tensor = tf.subtract(Cov_tensor, I_tensor)
        assert len(diff_tensor.get_shape().as_list()) == 2
        assert diff_tensor.get_shape().as_list()[0] == diff_tensor.get_shape().as_list()[1] == out_dim
        # reduce_mean: averging the elements
        tmp_loss = tf.multiply(tf.constant(0.5, dtype=tf.float32), tf.reduce_mean(tf.multiply(diff_tensor, diff_tensor)))
    elif extra_fc_type == 2:
        # soft W decorrelation (L1 loss)
        # averging sample num
        Cov_tensor = tf.div(tf.matmul(W_tensor, W_tensor, transpose_a=True), tf.constant(in_dim - 1, dtype=tf.float32))
        Corr_abs_tensor = tf.abs(Cov_tensor)
        assert len(Corr_abs_tensor.get_shape().as_list()) == 2
        assert Corr_abs_tensor.get_shape().as_list()[0] == Corr_abs_tensor.get_shape().as_list()[1] == out_dim
        # averging the elements
        tmp_loss = tf.div( tf.subtract(tf.reduce_sum(Corr_abs_tensor), tf.trace(Corr_abs_tensor)),
                           tf.constant(out_dim * (out_dim-1), dtype=tf.float32) )
    elif extra_fc_type == 3:
        # std W then soft decorrelation (L1 loss)
        mean, var = tf.nn.moments(W_tensor, [0], keep_dims=True)
        assert len(mean.get_shape().as_list()) == len(var.get_shape().as_list()) == 2
        assert mean.get_shape().as_list()[1] == var.get_shape().as_list()[1] == out_dim
        W_std_tensor = tf.div(tf.subtract(W_tensor, mean), tf.add(tf.sqrt(var), tf.constant(0.000001, dtype=tf.float32)))
        # averging sample num
        Cov_tensor = tf.div(tf.matmul(W_std_tensor, W_std_tensor, transpose_a=True), tf.constant(in_dim - 1, dtype=tf.float32))
        Corr_abs_tensor = tf.abs(Cov_tensor)
        assert len(Corr_abs_tensor.get_shape().as_list()) == 2
        assert Corr_abs_tensor.get_shape().as_list()[0] == Corr_abs_tensor.get_shape().as_list()[1] == out_dim
        # averging the elements
        tmp_loss = tf.div(tf.subtract(tf.reduce_sum(Corr_abs_tensor), tf.trace(Corr_abs_tensor)),
                          tf.constant(out_dim * (out_dim - 1), dtype=tf.float32))
    else:
        raise ValueError('Unrecognized extra_fc_type: {}'.format(extra_fc_type))

    loss = tf.multiply(tf.constant(extra_fc_W_decay, dtype=tf.float32), tmp_loss)
    tf.add_to_collection(loss_collection, loss)

@slim.add_arg_scope
def stack_blocks_dense(net, blocks, output_stride=None,
                       outputs_collections=None):
  """Stacks ResNet `Blocks` and controls output feature density.

  First, this function creates scopes for the ResNet in the form of
  'block_name/unit_1', 'block_name/unit_2', etc.

  Second, this function allows the user to explicitly control the ResNet
  output_stride, which is the ratio of the input to output spatial resolution.
  This is useful for dense prediction tasks such as semantic segmentation or
  object detection.

  Most ResNets consist of 4 ResNet blocks and subsample the activations by a
  factor of 2 when transitioning between consecutive ResNet blocks. This results
  to a nominal ResNet output_stride equal to 8. If we set the output_stride to
  half the nominal network stride (e.g., output_stride=4), then we compute
  responses twice.

  Control of the output feature density is implemented by atrous convolution.

  Args:
    net: A `Tensor` of size [batch, height, width, channels].
    blocks: A list of length equal to the number of ResNet `Blocks`. Each
      element is a ResNet `Block` object describing the units in the `Block`.
    output_stride: If `None`, then the output will be computed at the nominal
      network stride. If output_stride is not `None`, it specifies the requested
      ratio of input to output spatial resolution, which needs to be equal to
      the product of unit strides from the start up to some level of the ResNet.
      For example, if the ResNet employs units with strides 1, 2, 1, 3, 4, 1,
      then valid values for the output_stride are 1, 2, 6, 24 or None (which
      is equivalent to output_stride=24).
    outputs_collections: Collection to add the ResNet block outputs.

  Returns:
    net: Output tensor with stride equal to the specified output_stride.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  # The current_stride variable keeps track of the effective stride of the
  # activations. This allows us to invoke atrous convolution whenever applying
  # the next residual unit would result in the activations having stride larger
  # than the target output_stride.
  current_stride = 1

  # The atrous convolution rate parameter.
  rate = 1

  for block in blocks:
    with tf.variable_scope(block.scope, 'block', [net]) as sc:
      for i, unit in enumerate(block.args):
        if output_stride is not None and current_stride > output_stride:
          raise ValueError('The target output_stride cannot be reached.')

        with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
          unit_depth, unit_depth_bottleneck, unit_stride = unit

          # If we have reached the target output_stride, then we need to employ
          # atrous convolution with stride=1 and multiply the atrous rate by the
          # current unit's stride for use in subsequent layers.
          if output_stride is not None and current_stride == output_stride:
            net = block.unit_fn(net,
                                depth=unit_depth,
                                depth_bottleneck=unit_depth_bottleneck,
                                stride=1,
                                rate=rate)
            rate *= unit_stride

          else:
            net = block.unit_fn(net,
                                depth=unit_depth,
                                depth_bottleneck=unit_depth_bottleneck,
                                stride=unit_stride,
                                rate=1)
            current_stride *= unit_stride
      net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

  if output_stride is not None and current_stride != output_stride:
    raise ValueError('The target output_stride cannot be reached.')

  return net


# ex-final feats project and sum with final feat
@slim.add_arg_scope
def projecting_feats(net, gate_proj_type, proj_dim, gate_prop_down, is_training, avr_cc=0, outputs_collections=None):
    """
    projecting feats and/or deep (fc) features to the same dimension and add them up
    :param net: [batch_size, 1, 1, channel_num]
    :param gate_proj_type:
                        0: projecting gates to dim of deep feats
                        1: Projecting gates and deep feats to same dimension (proj_dim)
    :param proj_dim: joint dimension
    :param gate_prop_down: whether propagate down on gating activations
    :param outputs_collections:
    :return:
        averaged features
    """

    local_sc_name = 'pool5_feats_proj_add'

    # retrievaling gates using endpoints
    tmp_end_points = slim.utils.convert_collection_to_dict(outputs_collections)
    # pdb.set_trace()
    with tf.variable_scope(local_sc_name) as sc:
        net_add_feats = proj_feats_fn(net, gate_proj_type, proj_dim, gate_prop_down, tmp_end_points, avr_cc, is_training)
        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                net_add_feats)


def proj_feats_fn(net, gate_proj_type, proj_dim, gate_prop_down, tmp_end_points, avr_cc, is_training):
    # all output feats from different blocks are used
    print('Using gates from all blocks.')
    gate_feat_len = 32
    block_unit_pat_str = 'block[0-9]+?/unit_[0-9]+?'
    # sigmoid gates
    select_pat_str = r'resnet_v[1-2]_[0-9]+?/{}/bottleneck_v[1-2]$'.format(block_unit_pat_str)

    # check size: [batch_size, 1, 1, channel_num]
    assert len(net.get_shape().as_list()) == 4
    assert net.get_shape().as_list()[1] == 1
    assert net.get_shape().as_list()[2] == 1
    feat_channle_num = net.get_shape().as_list()[3]
    if gate_proj_type == 0:
        # deep feature not project, gate project to deep
        proj_dim = feat_channle_num
    elif gate_proj_type == 1:
        # both deep feats and gates feats project to proj_dim
        print('projecting deep feats to {}.'.format(proj_dim))
        net = proj_2Din4_bn_relu(net, proj_dim, sc_name='df_proj_layer')
    else:
        raise ValueError('Unrecognised projection type: {}'.format(gate_proj_type))
    assert len(net.get_shape().as_list()) == 4
    assert net.get_shape().as_list()[1] == 1
    assert net.get_shape().as_list()[2] == 1
    assert net.get_shape().as_list()[3] == proj_dim

    print('fetching the mid-level feats.')
    end_point_key_list = tmp_end_points.keys()
    selected_keys = []
    for _key in end_point_key_list:
        selected_list = re.findall(select_pat_str, _key)
        if len(selected_list) == 1:
            selected_keys.append(selected_list[0])

    feat_tensor_list = []
    # pdb.set_trace()
    ex_final_selected_keys = sorted(selected_keys)[:-1]
    assert len(selected_keys)-1 == len(ex_final_selected_keys)
    concate_len_check = 0
    for gate_key in sorted(ex_final_selected_keys):
        # reduce mean on spatial dimensions: axis = [1,2]
        feat_WH_tensor = tmp_end_points[gate_key]
        assert len(feat_WH_tensor.get_shape().as_list()) == 4
        tmp_f_dim = feat_WH_tensor.get_shape().as_list()[3]
        # mean over spatial
        feat_mean_tensor = tf.reduce_mean(feat_WH_tensor, [1, 2])
        assert len(feat_mean_tensor.get_shape().as_list()) == 2
        assert feat_mean_tensor.get_shape().as_list()[1] == tmp_f_dim
        concate_len_check += tmp_f_dim
        # pdb.set_trace()
        feat_tensor_list.append(feat_mean_tensor)

    tmp_mid_feats_tensor = tf.concat(feat_tensor_list, axis=1)
    # pdb.set_trace()
    # if concat_gate_sreg and is_training:
    #     gate_reg_loss(concat_gate_reg_type, tmp_gate_feat_tensor, 1.0)
    if gate_prop_down:
        mid_feats_tensor = tf.expand_dims(tf.expand_dims(tmp_mid_feats_tensor, axis=1), axis=1)
    else: # no propagate down for gate tensor
        mid_feats_tensor = tf.stop_gradient(tf.expand_dims(tf.expand_dims(tmp_mid_feats_tensor, axis=1), axis=1))
    # check size
    assert len(mid_feats_tensor.get_shape().as_list()) == 4
    mid_feats_len = mid_feats_tensor.get_shape().as_list()[3]
    assert mid_feats_len == concate_len_check
    # assert float(gates_feat_len)/gate_feat_len == int(gates_feat_len/gate_feat_len)

    print('projecting ex-final concate feats')

    ex_final_feat_tensor = proj_2Din4_bn_relu(mid_feats_tensor, proj_dim, sc_name='gate_proj_layer')
    assert len(ex_final_feat_tensor.get_shape().as_list()) == 4
    assert ex_final_feat_tensor.get_shape().as_list()[1] == 1
    assert ex_final_feat_tensor.get_shape().as_list()[2] == 1
    assert ex_final_feat_tensor.get_shape().as_list()[3] == proj_dim

    if avr_cc == 0:
        print('adding up')
        out_feat_tensor = tf.multiply(tf.add(net, ex_final_feat_tensor), tf.constant(0.5))
        assert len(out_feat_tensor.get_shape().as_list()) == 4
        assert out_feat_tensor.get_shape().as_list()[1] == 1
        assert out_feat_tensor.get_shape().as_list()[2] == 1
        assert out_feat_tensor.get_shape().as_list()[3] == proj_dim
    elif avr_cc == 1:
        print('Concatenating')
        out_feat_tensor = tf.concat([net, ex_final_feat_tensor], 3)
        assert len(out_feat_tensor.get_shape().as_list()) == 4
        assert out_feat_tensor.get_shape().as_list()[1] == 1
        assert out_feat_tensor.get_shape().as_list()[2] == 1
        assert out_feat_tensor.get_shape().as_list()[3] == proj_dim * 2
    else:
        raise ValueError('Undefined fusion method: {}'.format(avr_cc))
    return out_feat_tensor



def proj_2Din4_bn_relu(in_2Din4_tensor, proj_dim, sc_name=None, gate_net_weight_decay=0.0001):
    # 4D to 2D tensor
    in_2Din4_tensor = tf.squeeze(in_2Din4_tensor)
    assert len(in_2Din4_tensor.get_shape().as_list()) == 2

    with tf.variable_scope(sc_name, 'proj_layer') as sc:
        with slim.arg_scope([slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(gate_net_weight_decay),
                            weights_initializer=slim.xavier_initializer(),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm):

            out_tensor = slim.fully_connected(in_2Din4_tensor, proj_dim)

            assert len(out_tensor.get_shape().as_list()) == 2
            assert out_tensor.get_shape().as_list()[1] == proj_dim

            # return 4D tensor
            return tf.expand_dims(tf.expand_dims(out_tensor, axis=1), axis=1)







def resnet_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  """Defines the default ResNet arg scope.

  TODO(gpapan): The batch-normalization related default values above are
    appropriate for use in conjunction with the reference ResNet models
    released at https://github.com/KaimingHe/deep-residual-networks. When
    training ResNets from scratch, they might need to be tuned.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.

  Returns:
    An `arg_scope` to use for the resnet models.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      # The following implies padding='SAME' for pool1, which makes feature
      # alignment easier for dense prediction tasks. This is also used in
      # https://github.com/facebook/fb.resnet.torch. However the accompanying
      # code of 'Deep Residual Learning for Image Recognition' uses
      # padding='VALID' for pool1. You can switch to that choice by setting
      # slim.arg_scope([slim.max_pool2d], padding='VALID').
      with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc
