# this is a modification based on the train_image_classifier.py
# ==============================================================================
"""Generic training script that trains a ReID model using a given dataset."""

#######################################
## This file is to train DistributionNet based on Resnet-50 pretrained by Market-1501
# Note: This is step 2 of our method. The model is Resnet-distribution-50, and loss is Xent of mean and samples drawn from distributions. The optional losses include entropy loss and two versions of across instance losses
########################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import time

from tensorflow.python.ops import control_flow_ops
import dataset_factory
import model_deploy
from nets import nets_factory

import tensorflow.contrib.slim as slim

from utils import config_and_print_log, _configure_learning_rate, _configure_optimizer, _get_variables_to_train, _get_init_fn, get_img_func, build_graph, get_pair_type


##############
# FLags may be usually changed #
##############

tf.app.flags.DEFINE_string('model_name', 'resnet_v1_distributions_50', 'The name of the architecture to train.')

tf.app.flags.DEFINE_boolean('entropy_loss', False, 'uncertainty loss')
tf.app.flags.DEFINE_integer('max_number_of_steps', 60200, 'The maximum number of training steps.')
tf.app.flags.DEFINE_string('target', 'market', 'For name of model')
tf.app.flags.DEFINE_boolean('standard', False, 'For name of model')

tf.app.flags.DEFINE_string('set', 'bounding_box_train', "subset under current dataset")
tf.app.flags.DEFINE_integer('boot_weight', 1.0, 'cross entropy loss weight')

tf.app.flags.DEFINE_float('sampled_ce_loss_weight', 0.1, 'loss weight for xent of drawn samples.')
tf.app.flags.DEFINE_integer('sample_number', 1, 'the number of samples drawn from distribution.')

tf.app.flags.DEFINE_boolean('resume_train', True, 'when set to true, resume training from current train dir or load dir.')

tf.app.flags.DEFINE_string('dataset_name', 'Market', 'The name of the Person ReID dataset to load.')

tf.app.flags.DEFINE_string('dataset_dir', './Market/',
                           'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string('checkpoint_path2', '/import/vision-ephemeral/ty303/result/resnet_v1_50_emdNone_targetmarket_standard',
                           'The path to a checkpoint from which to fine-tune.')


tf.app.flags.DEFINE_float('learning_rate', 0.00005, 'Initial learning rate.')
tf.app.flags.DEFINE_float('end_learning_rate', 0.00001, 'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_string('trainable_scopes', 'resnet_v1_50/Distributions, resnet_v1_50/logits, resnet_v1_50/block4/', 'Comma-separated list of scopes to filter the set of variables to train. By default, None would train all the variables.')

tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', ['Distributions'], 'Comma-separated list of scopes of variables to exclude when restoring from a checkpoint.')


#####################
###The following flags are fixed all the time
#####################

tf.app.flags.DEFINE_boolean('use_clf', True, 'Add classification (identification) loss to the network.')
tf.app.flags.DEFINE_string('master', '', 'The address of the TensorFlow master to use.')
tf.app.flags.DEFINE_string('train_dir', './result',
                           'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_string('sub_dir', '', 'Subdirectory to identify the sv dir')
tf.app.flags.DEFINE_integer('num_clones', 1, 'Number of model clones to deploy.')
tf.app.flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones.')
tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')
tf.app.flags.DEFINE_integer('num_ps_tasks', 0, 'The number of parameter servers. If the value is 0, then the '
                                               'parameters are handled locally by the worker.')

# readers number
tf.app.flags.DEFINE_integer('num_readers', 4, 'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer('num_preprocessing_threads', 4, 'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer('log_every_n_steps', 100, 'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer('task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float('weight_decay', 0.0001, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'The name of the optimizer, one of "adadelta", "adagrad", "adam", '
                                                '"ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float('adam_beta1', 0.9, 'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float('adam_beta2', 0.999, 'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('opt_epsilon', 1e-5, 'Epsilon term for the optimizer.')


#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exponential', 'Specifies how the learning rate is decayed. One'
                                                                      ' of "fixed", "exponential", or "polynomial"')

tf.app.flags.DEFINE_float('label_smoothing', 0.1, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.95, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0, 'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_bool('sync_replicas', False, 'Whether or not to synchronize the replicas during training.')
tf.app.flags.DEFINE_integer('replicas_to_aggregate', 1, 'The Number of gradients to collect before updating params.')
tf.app.flags.DEFINE_float('moving_average_decay', None, 'The decay to use for the moving average. If left as None, '
                                                        'then moving averages are not used.')

#######################
# Dataset Flags #
#######################
tf.app.flags.DEFINE_string('source', None, 'detected, labeled, mixed')
tf.app.flags.DEFINE_integer('split_num', None, '0-19')
tf.app.flags.DEFINE_integer('cam_num', None, '6 cams or 10 cams.')
tf.app.flags.DEFINE_boolean('hd_data', False, 'using high resolution image data for training.')
tf.app.flags.DEFINE_integer('labels_offset', 0, 'An offset for the labels in the dataset. This flag is primarily used '
                                                'to evaluate the VGG and ResNet architectures which do not use a '
                                                'background class for the ImageNet dataset.')
tf.app.flags.DEFINE_string('model_scope', '', 'The name scope of given model.')
tf.app.flags.DEFINE_string('preprocessing_name', None, 'The name of the preprocessing to use. If left as `None`, then '
                                                       'the model_name flag is used.')
tf.app.flags.DEFINE_integer('batch_size', 32, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('batch_k', 4, 'The number of samples for each class (identity) in each batch.')
tf.app.flags.DEFINE_string('shuffle_order', 'T', 'whether shuffle the batch order. T for Ture; F for False')
tf.app.flags.DEFINE_integer('aug_mode', 3, 'data augumentation(1,2,3)')
tf.app.flags.DEFINE_boolean('rand_erase', False, 'random erasing the image to augment the data')
tf.app.flags.DEFINE_integer('test_mode', 1, 'testing 1: central crop 2: (coner crops + central crop +) flips')
tf.app.flags.DEFINE_integer('train_image_height', 256, 'Crop Height')
tf.app.flags.DEFINE_integer('train_image_width', 128, 'Crop Width')
tf.app.flags.DEFINE_integer('summary_snapshot_steps', 20000, 'Summary save steps.')
tf.app.flags.DEFINE_integer('model_snapshot_steps', 10000, 'Model save steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string('checkpoint_path', None, 'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_boolean('imagenet_pretrain', True, 'Using imagenet pretrained model to initialise.')
tf.app.flags.DEFINE_boolean('ignore_missing_vars', False, 'When restoring a checkpoint would ignore missing variables.')

###############
# Other Flags #
###############
tf.app.flags.DEFINE_boolean('log_device_placement', False, "Whether to log device placement.")

FLAGS = tf.app.flags.FLAGS

##############################################
#          Main Training Fuction             #
##############################################

def main(_):

    if not os.path.isdir(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)

    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    if not FLAGS.aug_mode:
        raise ValueError('aug_mode need to be speficied.')

    if (not FLAGS.train_image_height) or (not FLAGS.train_image_width):
        raise ValueError('The image height and width must be define explicitly.')

    if FLAGS.hd_data:
        if FLAGS.train_image_height != 400 or FLAGS.train_image_width != 200:
            FLAGS.train_image_height, FLAGS.train_image_width = 400, 200
            print("set the image size to (%d, %d)" % (400, 200))

    # config and print log
    config_and_print_log(FLAGS)

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        #######################
        # Config model_deploy #
        #######################
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=FLAGS.task,
            num_replicas=FLAGS.worker_replicas,
            num_ps_tasks=FLAGS.num_ps_tasks)

        # Create global_step
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        #####################################
        # Select the preprocessing function #
        #####################################
        img_func = get_img_func()

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.DataLoader(FLAGS.model_name, FLAGS.dataset_name, FLAGS.dataset_dir, FLAGS.set, FLAGS.hd_data, img_func,
                                             FLAGS.batch_size, FLAGS.batch_k, FLAGS.max_number_of_steps, get_pair_type())

        ######################
        # Select the network #
        ######################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            weight_decay=FLAGS.weight_decay,
            is_training=True,
            sample_number= FLAGS.sample_number
        )

        ####################
        # Define the model #
        ####################
        def clone_fn(tf_batch_queue):
            return build_graph(tf_batch_queue, network_fn)

        clones = model_deploy.create_clones(deploy_config, clone_fn, [dataset.tf_batch_queue])
        first_clone_scope = deploy_config.clone_scope(0)
        # Gather update_ops from the first clone. These contain, for example,
        # the updates for the batch_norm variables created by network_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        # Add summaries for end_points.
        end_points = clones[0].outputs

        # Add summaries for losses.
        loss_dict = {}
        for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
            if loss.name == 'softmax_cross_entropy_loss/value:0':
                loss_dict['clf'] = loss
            elif 'softmax_cross_entropy_loss' in loss.name:
                loss_dict['sample_clf_'+str(loss.name.split('/')[0].split('_')[-1])] = loss
            elif 'entropy' in loss.name:
                loss_dict['entropy'] = loss
            else:
                raise Exception('Loss type error')

        #################################
        # Configure the moving averages #
        #################################
        if FLAGS.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None

        #########################################
        # Configure the optimization procedure. #
        #########################################
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = _configure_learning_rate(dataset.num_samples, global_step, FLAGS)
            optimizer = _configure_optimizer(learning_rate)

        if FLAGS.sync_replicas:
            # If sync_replicas is enabled, the averaging will be done in the chief
            # queue runner.
            optimizer = tf.train.SyncReplicasOptimizer(
                opt=optimizer,
                replicas_to_aggregate=FLAGS.replicas_to_aggregate,
                variable_averages=variable_averages,
                variables_to_average=moving_average_variables,
                replica_id=tf.constant(FLAGS.task, tf.int32, shape=()),
                total_num_replicas=FLAGS.worker_replicas)
        elif FLAGS.moving_average_decay:
            # Update ops executed locally by trainer.
            update_ops.append(variable_averages.apply(moving_average_variables))

        # Variables to train.
        variables_to_train = _get_variables_to_train()

        #  and returns a train_tensor and summary_op
        # total_loss is the sum of all LOSSES and REGULARIZATION_LOSSES in tf.GraphKeys
        total_loss, clones_gradients = model_deploy.optimize_clones(
            clones,
            optimizer,
            var_list=variables_to_train)

        # Create gradient updates.
        grad_updates = optimizer.apply_gradients(clones_gradients,
                                                 global_step=global_step)
        update_ops.append(grad_updates)

        update_op = tf.group(*update_ops)
        train_tensor = control_flow_ops.with_dependencies([update_op], total_loss,
                                                          name='train_op')

        train_tensor_list = [train_tensor]
        format_str = 'step %d, loss = %.2f'

        for loss_key in sorted(loss_dict.keys()):
            train_tensor_list.append(loss_dict[loss_key])
            format_str += (', %s_loss = ' % loss_key + '%.8f')

        format_str += ' (%.1f examples/sec; %.3f sec/batch)'

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=1)
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')

        ###########################
        # Kicks off the training. #
        ###########################
        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # load pretrained weights
        if FLAGS.checkpoint_path is not None:
            print("Load the pretrained weights")
            weight_ini_fn = _get_init_fn()
            weight_ini_fn(sess)
        else:
            print("Train from the scratch")

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        # for step in xrange(FLAGS.max_number_of_steps):
        for step in xrange(FLAGS.max_number_of_steps + 1):
            start_time = time.time()

            loss_value_list = sess.run(train_tensor_list, feed_dict=dataset.get_feed_dict())

            duration = time.time() - start_time

            # assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % FLAGS.log_every_n_steps == 0:
                # num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                # sec_per_batch = duration / FLAGS.num_gpus
                sec_per_batch = duration

                print(format_str % tuple([step] + loss_value_list + [examples_per_sec, sec_per_batch]))

            # Save the model checkpoint periodically.
            # if step % FLAGS.model_snapshot_steps == 0 or (step + 1) == FLAGS.max_number_of_steps:
            if step % FLAGS.model_snapshot_steps == 0:
                saver.save(sess, checkpoint_path, global_step=step)

        print('OK...')


if __name__ == '__main__':
    tf.app.run()
