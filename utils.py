# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SketchRNN data loading and image manipulation utilities."""

import os
import h5py
import tensorflow as tf

from tensorflow.python.ops.losses.losses_impl import Reduction, compute_weighted_loss
from tensorflow.python.framework import ops
import numpy as np

tfd = tf.contrib.distributions
slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

try:
    from tabulate import tabulate
except:
    print "tabulate lib not installed"


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately

    def flush(self) :
        for f in self.files:
            f.flush()


def config_and_print_log(FLAGS):
    _config_pretrain_model(FLAGS)
    log_dir = FLAGS.train_dir
    sub_dir_list = [FLAGS.sub_dir]
    if not FLAGS.use_clf:
        sub_dir_list.append('nclf')
    if FLAGS.checkpoint_path is None:
        sub_dir_list.append('ts')
    if FLAGS.rand_erase:
        sub_dir_list.append('re')
    if FLAGS.hd_data:
        sub_dir_list.append('hd')
    FLAGS.sub_dir = '_'.join(sub_dir_list)
    if FLAGS.dataset_name.lower() != 'market':
        FLAGS.sub_dir += '_' + FLAGS.dataset_name.lower()
    if FLAGS.sub_dir:
        FLAGS.train_dir = os.path.join(FLAGS.train_dir, FLAGS.sub_dir)
    if FLAGS.resume_train:
        FLAGS.train_dir = FLAGS.train_dir+'_con'
    if FLAGS.sampled_ce_loss_weight:
        FLAGS.train_dir = FLAGS.train_dir+'_scelw'+str(FLAGS.sampled_ce_loss_weight)
    if FLAGS.sample_number:
        FLAGS.train_dir = FLAGS.train_dir+'_sample'+str(FLAGS.sample_number)
    if FLAGS.target:
        FLAGS.train_dir = FLAGS.train_dir + '_target'+FLAGS.target
    if '_0.' in FLAGS.set:
        FLAGS.train_dir = FLAGS.train_dir + '_noise_'+FLAGS.set.split('train_')[1]
    if FLAGS.standard:
        FLAGS.train_dir = FLAGS.train_dir+'_standard'
    if FLAGS.entropy_loss:
        FLAGS.train_dir = FLAGS.train_dir+'_entropy'

    log_prefix = 'logs/' + FLAGS.dataset_name.lower() + '_%s_' % FLAGS.sub_dir
    log_prefix = os.path.join(log_dir, log_prefix)
    print_log(log_prefix, FLAGS)


def config_eval_ckpt_path(FLAGS, flag=1):
    _config_pretrain_model(FLAGS, is_train=False)
    full_ckpt_path = FLAGS.sub_dir
    return full_ckpt_path


def _config_pretrain_model(FLAGS, is_train=True):
    pretrian_dir = './pretrained_model'
    # pretrained_model = {'resnet':'resnet_v1_50.ckpt', 'inceptionv1':'inception_v1.ckpt', 'mobilenet':'mobile_net.ckpt'}
    pretrained_model = {'resnet_v1_50':'resnet_v1_50.ckpt', 'resnet_v2':'resnet_v2_50.ckpt', 'resnet_v1_distributions_50':'resnet_v1_50.ckpt', 'resnet_v1_distributions_baseline_50':'resnet_v1_50.ckpt',}
    model_scopes = {'resnet_v1_50': 'resnet_v1_50', 'resnet_v2': 'resnet_v2_50', 'resnet_v1_distributions_50': 'resnet_v1_50', 'resnet_v1_distributions_baseline_50': 'resnet_v1_50'}
    checkpoint_exclude_scopes = {'resnet_v1': ['logits', 'concat_comb', 'fusion'], 'resnet_v1_distributions_50': [], 'resnet_v1_distributions_baseline_50': [],}
    shared_checkpoint_exclude_scopes = ['verifier']
    for model_key in pretrained_model.keys():
        if model_key == FLAGS.model_name:
            if FLAGS.imagenet_pretrain and is_train:
                FLAGS.checkpoint_path = os.path.join(pretrian_dir, pretrained_model[model_key])
            if len(FLAGS.sub_dir) == 0:
                FLAGS.sub_dir = model_key
                print "Set sub_dir to %s" % model_key
            if len(FLAGS.model_scope) == 0:
                FLAGS.model_scope = model_scopes[model_key]
                print "Set model scope to %s" % model_scopes[model_key]
            if len(FLAGS.checkpoint_exclude_scopes) == 0 and is_train:
                FLAGS.checkpoint_exclude_scopes = checkpoint_exclude_scopes[model_key]
                FLAGS.checkpoint_exclude_scopes.extend(shared_checkpoint_exclude_scopes)
                print "Set checkpoint exclude scopes to :", checkpoint_exclude_scopes[model_key]


def _configure_learning_rate(num_samples_per_epoch, global_step, FLAGS):
    """Configures the learning rate.

    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.

    Returns:
      A `Tensor` representing the learning rate.

    Raises:
      ValueError: if
    """
    decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)
    if FLAGS.sync_replicas:
        decay_steps /= FLAGS.replicas_to_aggregate

    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.

    Args:
      learning_rate: A scalar or `Tensor` learning rate.

    Returns:
      An instance of an optimizer.

    Raises:
      ValueError: if FLAGS.optimizer is not recognized.
    """

    if FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.opt_epsilon)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer


def _get_init_fn():
    """Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
      An init function run by the supervisor.
    """
    if FLAGS.checkpoint_path is None:
        return None

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    #if tf.train.latest_checkpoint(FLAGS.train_dir) and FLAGS.resume_train:
     #   print('Ignoring --checkpoint_path because a checkpoint already exists in %s'% FLAGS.train_dir)
      #  return None

    exclusions = []
    model_scope = FLAGS.model_scope
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [model_scope + '/' + scope.strip() for scope in FLAGS.checkpoint_exclude_scopes]
    exclusions.append('instance/')

    if FLAGS.resume_train:
        FLAGS.checkpoint_path = FLAGS.checkpoint_path2

    # TODO(sguada) variables.filter_variables()


    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
            #print var

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    print('Fine-tuning from %s' % checkpoint_path)

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=FLAGS.ignore_missing_vars)


def _get_variables_to_train():
    """Returns a list of variables to train.

    Returns:
      A list of variables to train by the optimizer.
    """
    tmp = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    if FLAGS.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def print_log(log_prefix, FLAGS):
    # print header
    print "==============================================="
    print "Trainning ", FLAGS.model_name, " in this framework"
    print "==============================================="

    print "Tensorflow flags:"

    flags_list = []
    for attr, value in sorted(FLAGS.__flags.items()):
        flags_list.append(attr)
    FLAGS.saved_flags = " ".join(flags_list)

    flag_table = {}
    flag_table['FLAG_NAME'] = []
    flag_table['Value'] = []
    flag_lists = FLAGS.saved_flags.split()
    # print self.FLAGS.__flags
    for attr in flag_lists:
        if attr not in ['saved_flags', 'net_name', 'log_root']:
            flag_table['FLAG_NAME'].append(attr.upper())
            flag_table['Value'].append(getattr(FLAGS, attr))
    flag_table['FLAG_NAME'].append('HOST_NAME')
    flag_table['Value'].append(os.uname()[1].split('.')[0])
    try:
        print tabulate(flag_table, headers="keys", tablefmt="fancy_grid").encode('utf-8')
    except:
        for attr in flag_lists:
            print "attr name, ", attr.upper()
            print "attr value, ", getattr(FLAGS, attr)


def get_img_func(is_training=True):
    from ReID_preprocessing import preprocessing_factory
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        aug_mode=FLAGS.aug_mode,
        test_mode=FLAGS.test_mode,
        is_training=is_training,
        rand_erase=FLAGS.rand_erase,
    )

    def callback(images):
        return image_preprocessing_fn(images, FLAGS.train_image_height, FLAGS.train_image_width)

    return callback


def loss_entropy(
    mu, sig, weights=0.001, label_smoothing=0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):

    with ops.name_scope(scope, "entropy_loss",
                        (mu, sig, weights)) as scope:

        sigma_avg = 5
        threshold = np.log(sigma_avg) + (1 + np.log(2 * np.pi)) / 2
        losses = tf.reduce_mean(tf.nn.relu(threshold-tfd.MultivariateNormalDiagWithSoftplusScale(loc=mu, scale_diag=sig).entropy()/2048))

    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)


def build_graph(tf_batch_queue, network_fn):

    images, labels = tf_batch_queue[:2]

    if 'distribution' in FLAGS.model_name and 'baseline' not in FLAGS.model_name:
        logits, logits2, end_points = network_fn(images)
    else:
        logits, end_points = network_fn(images)

    #############################
    # Specify the loss function #
    #############################
    if FLAGS.use_clf:
        if 'AuxLogits' in end_points:
            tf.losses.softmax_cross_entropy(
                logits=end_points['AuxLogits'], onehot_labels=labels,
                label_smoothing=FLAGS.label_smoothing, weights=0.4, scope='aux_loss')
        tf.losses.softmax_cross_entropy(
            logits=logits, onehot_labels=labels,
            label_smoothing=FLAGS.label_smoothing, weights=FLAGS.boot_weight)
        if 'distribution' in FLAGS.model_name and 'baseline' not in FLAGS.model_name:
            for logits_ in logits2:
                tf.losses.softmax_cross_entropy(
                    logits=logits_, onehot_labels=labels,
                    label_smoothing=FLAGS.label_smoothing, weights=FLAGS.sampled_ce_loss_weight)

    if FLAGS.entropy_loss:
        mu = end_points['PreLogits_mean']
        sig = end_points['PreLogits_sig']
        loss_entropy(mu, sig)

    return end_points


def get_pair_type(is_training=True):
    if is_training:
        pair_type = 'single'
    else:
        pair_type = 'eval'
    return pair_type

