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
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import numpy as np

import tensorflow as tf

import dataset_factory
from nets import nets_factory

import tensorflow.contrib.slim as slim

from utils import config_eval_ckpt_path, get_img_func, get_pair_type

model = 'resnet_v1_50'

tf.app.flags.DEFINE_string('sub_dir', '', 'Subdirectory to identify the sv dir')

tf.app.flags.DEFINE_string('dataset_dir', '',
                           'The directory where the dataset files are stored.')


tf.app.flags.DEFINE_string('dataset_name', '', 'The name of the Person ReID dataset to load.')

tf.app.flags.DEFINE_string('feat_layer', 'PreLogits', 'The name of the feature layer.')

tf.app.flags.DEFINE_integer('batch_size', 100, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('batch_k', 4, 'The number of samples for each class (identity) in each batch.')
tf.app.flags.DEFINE_integer('max_num_batches', None, 'Max number of batches to evaluate by default use all.')
tf.app.flags.DEFINE_string('master', '', 'The address of the TensorFlow master to use.')
tf.app.flags.DEFINE_string('checkpoint_path', '/import/vision-ephemeral/ty303/result',
                           'The directory where the model was written to or an absolute path to a checkpoint file.')
#tf.app.flags.DEFINE_string('checkpoint_path', '/homes/ty303/Downloads/slim_reid_h5/result',
#                           'The directory where the model was written to or an absolute path to a checkpoint file.')

tf.app.flags.DEFINE_string('eval_dir', '',
                           'Directory where the results are saved to.')
#tf.app.flags.DEFINE_string('eval_dir', '/homes/ty303/Downloads/slim_reid_h5/result',
#                           'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer('num_preprocessing_threads', 4, 'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer('GAN_Pose_v', None, 'The GAN Pose Version.')
tf.app.flags.DEFINE_string('set', None, "train, all_test_prb, all_test_gal, randOne_test_prb, randOne_test_gal")
tf.app.flags.DEFINE_integer('pose_n', None, 'The Pose to be extracted')
tf.app.flags.DEFINE_string('source', None, 'detected, labeled, mixed')
tf.app.flags.DEFINE_integer('split_num', None, '0-19')
tf.app.flags.DEFINE_integer('cam_num', None, '6 cams or 10 cams.')
tf.app.flags.DEFINE_integer('test_step', None, 'The step of the tested model.')
tf.app.flags.DEFINE_integer('max_number_of_steps', 60000, 'The maximum number of training steps.')

# tf.app.flags.DEFINE_string(
#     'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_boolean('hd_data', False, 'using high resolution image data for training.')
tf.app.flags.DEFINE_integer('labels_offset', 0, 'An offset for the labels in the dataset. This flag is primarily used '
                                                'to evaluate the VGG and ResNet architectures which do not use a '
                                                'background class for the ImageNet dataset.')
tf.app.flags.DEFINE_string('model_name', model, 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string('model_scope', '', 'The name scope of given model.')
# tf.app.flags.DEFINE_string('feat_layer', 'layer_19', 'The name of the feature layer.')
tf.app.flags.DEFINE_string('preprocessing_name', None, 'The name of the preprocessing to use. If left as `None`, then '
                                                       'the model_name flag is used.')
tf.app.flags.DEFINE_float('moving_average_decay', None, 'The decay to use for the moving average.If left as None, then '
                                                        'moving averages are not used.')

# tf.app.flags.DEFINE_integer(
#     'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_integer('aug_mode', 3, 'data augumentation(1,2,3)')
tf.app.flags.DEFINE_boolean('rand_erase', False, 'random erasing the image to augment the data')
tf.app.flags.DEFINE_integer('test_mode', 1, 'testing 1: central crop 2: (coner crops + central crop +) flips')
tf.app.flags.DEFINE_integer('train_image_height', 256, 'Crop Height')
tf.app.flags.DEFINE_integer('train_image_width', 128, 'Crop Width')


##############
# Loss FLags #
##############

tf.app.flags.DEFINE_boolean('use_clf', True, 'Add classification (identification) loss to the network.')

###############
# Other Flags #
###############
tf.app.flags.DEFINE_boolean('log_device_placement', False,"Whether to log device placement.")
tf.app.flags.DEFINE_boolean('imagenet_pretrain', True, 'Using imagenet pretrained model to initialise.')
tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', '', 'Comma-separated list of scopes of variables to exclude '
                                                              'when restoring from a checkpoint.')

FLAGS = tf.app.flags.FLAGS


def feat_aggregate(tmp_feats, image_list_len):
    if FLAGS.test_mode == 1:
        assert FLAGS.batch_size == tmp_feats.shape[0]
        return tmp_feats
    elif FLAGS.test_mode == 2:
        assert FLAGS.batch_size * image_list_len == tmp_feats.shape[0]
        assert len(tmp_feats.shape) == 2
        tmp_list = []
        for i in range(FLAGS.batch_size):
            same_p_feat_idx_list = range(i, FLAGS.batch_size * image_list_len, FLAGS.batch_size)
            assert len(same_p_feat_idx_list) == image_list_len
            same_p_feat_idx_arr = np.asarray(same_p_feat_idx_list)
            same_p_feats = tmp_feats[same_p_feat_idx_arr, :]
            assert same_p_feats.shape[0] == image_list_len
            tmp_list.append(np.mean(same_p_feats, axis=0))
        agged_feats = np.vstack(tmp_list)
        assert agged_feats.shape[0] == FLAGS.batch_size
        assert agged_feats.shape[1] == tmp_feats.shape[1]
        return agged_feats
    else:
        raise ValueError('No such test_mode')


def _extract_feats(Restorer, endpoints, feat_layer_name, num_samples, images, labels, dataset):
    # ImUIDs: guarantees all the testing examples are extracted only once.
    # f_names: file names
    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
        num_batches = FLAGS.max_num_batches
    else:
        # This ensures that we make a single pass over all of the data.
        num_batches = int(math.ceil(num_samples / float(FLAGS.batch_size)))

    tf.logging.info('Features from Layer: %s' % feat_layer_name)

    if feat_layer_name == 'sample_dist':
        activations_op = endpoints[feat_layer_name]
    elif ',' in feat_layer_name:
        activations_op = []
        tmp = feat_layer_name.split(',')
        for tmps in tmp:
            activations_op.append(endpoints[tmps])
    else:
        activations_op = tf.squeeze(endpoints[feat_layer_name])

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement,
    #    gpu_options=gpu_options
    ))
    sess.run(init)

    # config the directory to load the ckpt
    full_ckpt_path = config_eval_ckpt_path(FLAGS,flag=0)
    print(full_ckpt_path)

    if tf.gfile.IsDirectory(full_ckpt_path):
        checkpoint_path = tf.train.latest_checkpoint(full_ckpt_path)
    else:
        checkpoint_path = full_ckpt_path

    if checkpoint_path:
        global_step = checkpoint_path.split('/')[-1].split('-')[-1]
        print('Successfully loaded model from %s at step=%s.' % (checkpoint_path, global_step))
        FLAGS.test_step = global_step
    else:
        raise ValueError('No checkpoint file found at %s' % FLAGS.checkpoint_path)

    # restoring the weights
    Restorer.restore(sess, checkpoint_path)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    feats_list = []
    feats_sig_list = []
    labels_list = []
    f_names_list = []
    appeared_img_ids = []
    cameras_list = []

    tf_data, tf_label = dataset.tf_data, dataset.tf_label

    for step in range(num_batches * 2):  # since there is no shuffle, we need at most 2 round to get all the samples

        data_batch, label_batch, id_batch, name_batch = zip(*(dataset.gen_batches.next()))
        camera_batch = []
        for name in name_batch:
            #camera_batch.append((name[name.find('c') + 1]))
            if 'CUHK' not in FLAGS.dataset_dir:
                tmp = name.split('/')[-1]
                camera_batch.append(float(tmp[tmp.find('c')+1]))
            elif 'CUHK03' in FLAGS.dataset_dir:
                tmp = name.split('/')[-1].split('_')[2]
                camera_batch.append(float(tmp))
            else:
                tmp = name
                camera_batch.append((tmp[tmp.find('c') + 1]))
        camera_batch = tuple(camera_batch)
        #tmp_feats, sig_feats = sess.run(activations_op, feed_dict={tf_data: data_batch, tf_label: label_batch})
        tmp_feats = sess.run(activations_op, feed_dict={tf_data: data_batch, tf_label: label_batch})
        # check size
        data_batch, label_batch, id_batch = np.array(data_batch), np.array(label_batch), np.array(id_batch)
        assert id_batch.shape[0] == FLAGS.batch_size
        assert label_batch.shape[0] == FLAGS.batch_size
        sample_n, height, width, channel = data_batch.shape
        assert tmp_feats.shape[0] == sample_n
        # assert height == FLAGS.train_image_height
        # assert width == FLAGS.train_image_width
        assert channel == 3
        if FLAGS.test_mode == 1:
            assert sample_n == FLAGS.batch_size
        elif FLAGS.test_mode == 2:
            raise ValueError('Not implemented')
        else:
            raise ValueError('No such test_mode')
        # pdb.set_trace()

        # check the ImUIDs to be unique
        assert len(np.unique(id_batch)) == id_batch.shape[0]

        ImageUniIDs_list = id_batch.tolist()
        agg_feats = feat_aggregate(tmp_feats, 1)
        assert agg_feats.shape[0] == len(ImageUniIDs_list)
        for _idx in range(len(ImageUniIDs_list)):
            img_uid = ImageUniIDs_list[_idx]
            if img_uid not in appeared_img_ids:
                feats_list.append(agg_feats[_idx])
                labels_list.append(label_batch[_idx])
                f_names_list.append(name_batch[_idx])
                cameras_list.append(camera_batch[_idx])
                appeared_img_ids.append(img_uid)
        assert len(feats_list) == len(labels_list) == len(appeared_img_ids) == np.unique(appeared_img_ids).shape[0]
        if len(appeared_img_ids) == num_samples:
            break
    # check whether all unique samples are extracted.
    assert len(appeared_img_ids) == num_samples

    feats_ = np.array(feats_list)[0:num_samples, ::]
    labels_ = np.hstack(labels_list)[0:num_samples]
    cameras_ = np.hstack(cameras_list)[0:num_samples]
    names = f_names_list[0:num_samples]
    assert len(labels_) == num_samples
    assert feats_.shape[0] == num_samples

    # pdb.set_trace()
    # save the corresponding mat
    if FLAGS.dataset_name == 'Market':

        tmp = list(labels_).index(2)
        feats_ = feats_[tmp:]
        #feats_sig = feats_sig[tmp:]
        labels_ = labels_[tmp:]
        cameras_ = cameras_[tmp:]
        appeared_img_ids = appeared_img_ids[tmp:]
        names = names[tmp:]

    else:
        raise ValueError('Unrecognize dataset {}'.format(FLAGS.dataset_name))

    mat_dict = {'feats': feats_, 'labels': labels_, 'cameras': cameras_,
                'ImUIDs': np.asarray(appeared_img_ids, dtype=np.int), 'names': names}


    return mat_dict


def main(_):
    folder_path = './result'
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if 'distri' in f]# and (len(os.listdir(folder_path+'/'+f)) == 4)]# or len(os.listdir(folder_path+'/'+f)) == 7)]


    for file in files:

        FLAGS.sub_dir = file

        FLAGS.dataset_dir = './Market/'
        FLAGS.dataset_name = 'Market'

        sub_sets = ['bounding_box_test', 'query']

        #FLAGS.eval_dir += FLAGS.eval_dir + '/' + FLAGS.sub_dir + '_eval'

        feat_list = []

        for sub_set in sub_sets:
            FLAGS.set = sub_set
            print("Extracting features for %s" % sub_set)
            feat_list.append(extract_features())

        query = feat_list[-1]
        bounding_box_test = feat_list[0]

        q_feat_mean = query['feats']
        test_feat_mean = bounding_box_test['feats']

        distmat = np.zeros([q_feat_mean.shape[0], test_feat_mean.shape[0]])

        print(distmat.shape[0])

        print(distmat.shape[1])

        q_feat = q_feat_mean / np.sqrt(np.sum(np.power(q_feat_mean, 2), axis=-1))[:, None]
        test_feat = test_feat_mean / np.sqrt(np.sum(np.power(test_feat_mean, 2), axis=-1))[:, None]

        distmat += np.repeat(np.expand_dims(np.sum(np.power(q_feat,2),axis=-1), axis=1), test_feat.shape[0], axis=1)+ \
                  np.repeat(np.expand_dims(np.sum(np.power(test_feat, 2), axis=-1), axis=0), q_feat.shape[0], axis=0)

        distmat -= 2*np.matmul(q_feat,test_feat.transpose())

        print('average_mean: ' + str(distmat.mean()))

        cmc, map = eval_market(distmat, q_label = query['labels'], g_label = bounding_box_test['labels'], q_camera = query['cameras'], g_camera = bounding_box_test['cameras'], q_name=query['names'], g_name=bounding_box_test['names'])

        ranks = [1, 5, 10, 20]

        print("Results ----------")
        print("mAP: {:.8%}".format(map))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.8%}".format(r, cmc[r - 1]))
        print("------------------")

        dict = {}
        dict['map'] = str(map)
        dict['rank1'] = str(cmc[0])
        dict['rank5'] = str(cmc[4])
        dict['rank10'] = str(cmc[9])
        dict['rank20'] = str(cmc[19])

        import json
        with open(file+'/rank.txt', 'w') as f:
            json.dump(dict, f)




def eval_market(distmat, q_label, g_label, q_camera, g_camera, q_name, g_name, max_rank=20):
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat,axis=1)
    matches = (g_label[indices] == q_label[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        #print(q_idx)
        # get query pid and camid
        if 'Market' in FLAGS.dataset_dir or 'Duke' in FLAGS.dataset_dir or 'CUHK03' in FLAGS.dataset_dir:
            q_pid = q_label[q_idx]
            q_camid = q_camera[q_idx]

            order = indices[q_idx]
            remove = (g_label[order] == q_pid) & (g_camera[order] == q_camid)
            keep = np.invert(remove)

            # compute cmc curve
            orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        else:
            orig_cmc = matches[q_idx]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP



def extract_features():
    #if not os.path.isdir(FLAGS.eval_dir):
    #    os.makedirs(FLAGS.eval_dir)

    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    if not FLAGS.aug_mode:
        raise ValueError('aug_mode need to be speficied.')

    if (not FLAGS.train_image_height) or (not FLAGS.train_image_width):
        raise ValueError('The image height and width must be define explicitly.')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        #####################################
        # Select the preprocessing function #
        #####################################
        img_func = get_img_func(is_training=False)

        ######################
        # Select the dataset #
        ######################
        # testing pose extraction
       
        
        dataset = dataset_factory.DataLoader(FLAGS.model_name, FLAGS.dataset_name, FLAGS.dataset_dir, FLAGS.set, FLAGS.hd_data, img_func,
                                             FLAGS.batch_size, FLAGS.batch_k, FLAGS.max_number_of_steps,
                                             get_pair_type(is_training=False))

        ####################
        # Select the model #
        ####################
        # testing mode
        # class num is None, is_training=False
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=None,
            is_training=False)  # num_classes=None, should return pool5 features


        ####################
        # Define the model #
        ####################

        images, labels = dataset.tf_batch_queue[:2]

        logits, endpoints = network_fn(images)
       
        endpoints['pool5'] = logits

        #############################
        # code segment from _get_init_fn
        # get the variables_to_restore considering FLAGS.checkpoint_exclude_scopes
        exclusions = []
        if FLAGS.checkpoint_exclude_scopes:
            exclusions = [scope.strip()
                          for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

        # TODO(sguada) variables.filter_variables()
        variables_to_restore = []
        tmp = slim.get_model_variables()
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)

        labels = tf.squeeze(labels)

        Restorer = tf.train.Saver(variables_to_restore)

        feature_dict = _extract_feats(Restorer, endpoints, FLAGS.feat_layer, dataset.num_samples, images, labels, dataset)

        return feature_dict


if __name__ == '__main__':
    tf.app.run()

