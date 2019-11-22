import os
import h5py
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

def load_hdf5(fname):
    hf = h5py.File(fname, 'r')
    d = {key: np.array(hf.get(key)) for key in hf.keys()}
    hf.close()
    return d


class DataLoader(object):
    """Class for loading data."""

    def __init__(self, model, name, dataset_dir, _set, hd, img_func, batch_size, batch_k, num_iters, pair_type='single', openset=False, target_number=0):
        self.model = model
        self.dataset = name
        self.dataset_dir = dataset_dir
        self.subset = _set
        self.img_func = img_func
        self.batch_size = batch_size
        self.batch_k = batch_k
        self.batch_p = batch_size/batch_k
        self.pair_type = pair_type
        self.num_iters = num_iters
        self.hdf5_dir = get_dataset_dir(name, dataset_dir)
        self.hd_flag = hd
        self.target_number = target_number
        if openset:
            self.load_data(openset)
        else:
            self.load_data()
        if 'clean' in model:
            self.load_ref()
        self.get_pairs()
        self.set_epochs()
        self.creat_tf_data()
        self.process_tf_data()
        self.config_get_batch()

    def load_ref(self):
        file_path = '/import/vision-ephemeral/ty303/clean-net-master/util/'+self.dataset+'/ref.npy'
        self.ref_data = np.load(file_path)

        file_path = '/import/vision-ephemeral/ty303/clean-net-master/util/'+self.dataset+'/0.1_0_train.npy'
        temp = np.load(file_path)

        a = 1


    def load_data(self, openset=False):

        if self.hd_flag:
            self.hdf5_file = os.path.join(self.hdf5_dir, '%s_hd.h5' % self.subset)
        else:
            self.hdf5_file = os.path.join(self.hdf5_dir, '%s.h5' % self.subset)
        
        data_info = load_hdf5(self.hdf5_file)
        self.label = data_info['label']
        if openset:
            import random
            if self.target_number == 15:
                self.label = np.array(self.label)
                label_tmp = np.unique(self.label)
                rand_start_label = np.random.choice(label_tmp[:-16],1)
                #rand_start_label = np.random.choice(label_tmp[:-(int(0.01*len(self.label))+2)],1)
                #rand_start_label = random.randint(0, max(self.label))
                start = self.label.tolist().index(rand_start_label)
                #rand_end_label = label_tmp[label_tmp.tolist().index(rand_start_label)+int(0.01*len(self.label))+1]
                rand_end_label = label_tmp[label_tmp.tolist().index(rand_start_label)+15+1]
                end = self.label.tolist().index(rand_end_label)

            if self.target_number == 100:
                start = self.label.tolist().index(100)
                end = self.label.tolist().index(200)

            self.label = self.label[start:end]
            self.data = data_info['image_data'][start:end]

            self.pid = data_info['pid'][start:end]
            self.cid = data_info['cid'][start:end]
            self.image_id = data_info['id'][start:end]
            self.image_name = data_info['image_name'][start:end]
        else:
            self.data = data_info['image_data']
            self.pid = data_info['pid']
            self.cid = data_info['cid']
            self.image_id = data_info['id']
            self.image_name = data_info['image_name']

        self.num_classes = max(self.label) - min(self.label) + 1
        tmp = set(self.label)

        self.data_size, self.image_height, self.image_width, self.num_channel = self.data.shape
        self.num_samples = self.data_size # notice that num_samples is equal to number of training images

    def creat_tf_data(self):
        batch_size, height, width, num_channel = self.batch_size, self.image_height, self.image_width, self.num_channel
        self.tf_data = tf.placeholder(tf.float32, shape=(batch_size, height, width, num_channel), name="data")
        if 'clean' in self.model:
            self.tf_ref_data = tf.placeholder(tf.float32, shape=(batch_size, 2, 2048), name="ref_data")
        self.tf_label = tf.placeholder(tf.int32, shape=batch_size, name="label")
        self.tf_vlabel = tf.placeholder(tf.int32, shape=batch_size, name="vlabel")
        self.tf_vflag = tf.placeholder(tf.int32, shape=batch_size, name="vflag")
        if self.pair_type == 'pair':
            full_product_size = batch_size*batch_size/2
            self.tf_pair = tf.placeholder(tf.int32, shape=(batch_size, full_product_size/batch_size, 3), name="pair")
        if self.pair_type == 'trip':
            product_size = batch_size*batch_size/4
            self.tf_trip = tf.placeholder(tf.int32, shape=(batch_size, product_size/batch_size, 3), name="trip")
        if self.pair_type == 'hard':
            product_size = self.batch_p*self.batch_k*self.batch_k*(self.batch_p-1)
            self.tf_trip = tf.placeholder(tf.int32, shape=(batch_size, product_size/batch_size, 4), name="trip")

    def process_tf_data(self):
        self.tf_images = self.img_func(self.tf_data)
        self.tf_labels = slim.one_hot_encoding(self.tf_label, self.num_classes)
        if 'clean' in self.model:
            self.tf_batch_queue = [self.tf_images, self.tf_labels, self.tf_ref_data, self.tf_vlabel, self.tf_vflag] # input for network
            self.tf_batch_tuple = (self.tf_data, self.tf_label, self.tf_ref_data, self.tf_vlabel, self.tf_vflag) # input for feed dict
        elif self.pair_type in ['single', 'eval']:
            self.tf_batch_queue = [self.tf_images, self.tf_labels] # input for network
            self.tf_batch_tuple = (self.tf_data, self.tf_label) # input for feed dict
        elif self.pair_type == 'pair':
            self.tf_batch_queue = [self.tf_images, self.tf_labels, self.tf_pair]  # input for network
            self.tf_batch_tuple = (self.tf_data, self.tf_label, self.tf_pair)  # input for feed dict
        elif self.pair_type in ['trip', 'hard']:
            self.tf_batch_queue = [self.tf_images, self.tf_labels, self.tf_trip] # input for network
            self.tf_batch_tuple = (self.tf_data, self.tf_label, self.tf_trip) # input for feed dict
        else:
            raise  Exception('pair type error')

    def get_pairs(self):
        if self.pair_type in ['single', 'eval']:
            print "no need to load pairs"

        elif self.pair_type in ['pair', 'trip', 'hard']:
            min_width = self.batch_size/2
            if self.pair_type == 'hard':
                min_width = self.batch_p
            self.pairs, self.data_size = gen_data_pairs(self.label, min_width, self.batch_k, self.pair_type)
            print "set data size to the number of classes: %d" % self.data_size
        else:
            raise Exception('pair type error')

    def set_epochs(self):
        self.num_batches_per_epoch = int(np.ceil(self.data_size*1.0/self.batch_size))
        if self.pair_type in ['pair', 'trip']:
            self.num_epochs = int(np.ceil(self.num_iters*1.0/self.data_size))
        else:
            self.num_epochs = int(np.ceil(self.num_iters*1.0/self.num_batches_per_epoch))

    def config_get_batch(self):
        if self.pair_type == 'single':
            self.gen_batches = self.batch_iter()
        elif self.pair_type == 'pair':
            self.gen_batches = self.pair_batch_iter()
        elif self.pair_type == 'trip':
            self.gen_batches = self.trip_batch_iter()
        elif self.pair_type == 'hard':
            self.gen_batches = self.hard_trip_batch_iter()
        elif self.pair_type == 'eval':
            self.gen_batches = self.batch_iter_eval()
        else:
            raise Exception('Pair type error')

    def get_feed_dict(self):

        batch_tuple = zip(*(self.gen_batches.next()))
        feed_dict = {tfdata_elem: batch_elem for tfdata_elem, batch_elem in zip(self.tf_batch_tuple, batch_tuple)}
        # import pdb
        # pdb.set_trace()
        # print feed_dict[feed_dict.keys()[0]].shape
        return feed_dict

    def batch_iter(self):
        # Generates a batch iterator for a dataset with naive data and label format.
        batch_size = self.batch_size
        for epoch in range(self.num_epochs):
            # Shuffle the data at each epoch
            shuffle_indices = np.random.permutation(np.arange(self.data_size))
            for batch_id in range(self.num_batches_per_epoch):
                start_index = batch_id * batch_size
                indices = np.remainder(np.arange(start_index, start_index + batch_size), self.data_size)
                indices = shuffle_indices[indices]

                data_batch = self.data[indices, ::]
                label_batch = self.label[indices]
                if 'clean' in self.model:
                    ref_batch = self.ref_data[label_batch]
                    yield zip(data_batch, label_batch, ref_batch)
                else:
                    yield zip(data_batch, label_batch)


    def batch_iter_eval(self):
        # Generates a batch iterator for a dataset with naive data and label format.
        batch_size = self.batch_size
        for epoch in range(self.num_epochs):
            for batch_id in range(self.num_batches_per_epoch):
                start_index = batch_id * batch_size
                indices = np.remainder(np.arange(start_index, start_index + batch_size), self.data_size)

                data_batch = self.data[indices, ::]
                label_batch = self.label[indices]
                id_batch = self.image_id[indices]
                name_batch = self.image_name[indices, 0]
                yield zip(data_batch, label_batch, id_batch, name_batch)

    def pair_batch_iter(self):
        # generate a batch iterator with paired data
        batch_size = self.batch_size
        half_batch_size = batch_size/2
        product_size = half_batch_size*half_batch_size
        full_product_size = product_size + product_size
        pair_data = np.ones((full_product_size, 3), dtype=np.int)
        # pair_data[:product_size, 2] = 1
        pair_data[product_size:, 2] = 0
        for epoch in range(self.num_epochs):
            # shuffle trip array, equals to shuffle the data, but can keep the relation between data and hard trip list
            for batch_id in range(self.data_size):
                pos_inds = np.random.permutation(self.pairs['pos'][batch_id])[:half_batch_size]
                neg_class_ids = np.random.choice(np.arange(self.data_size), batch_size-half_batch_size)
                neg_inds = [np.random.choice(self.pairs['pos'][neg_class_id]) for neg_class_id in neg_class_ids]
                batch_inds = list(itertools.chain(pos_inds, neg_inds))

                pair_data[:product_size, 0] = self.pairs['p2p'][0]
                pair_data[:product_size, 1] = self.pairs['p2p'][1]
                pair_data[product_size:, 0] = self.pairs['p2p'][0]
                pair_data[product_size:, 1] = self.pairs['p2n'][1]

                indices = np.random.permutation(np.arange(full_product_size))
                # pair_indices = np.random.permutation(np.arange(batch_size))

                data_batch = self.data[batch_inds, ::]
                label_batch = self.label[batch_inds]
                pair_data_batch = np.reshape(pair_data[indices, ::], [batch_size, -1, 3])
                # pair_data_batch = np.reshape(pair_data, [batch_size, -1, 3])[pair_indices, ::]

                yield zip(data_batch, label_batch, pair_data_batch)

    def trip_batch_iter(self):
        # generate a batch iterator with paired data
        batch_size = self.batch_size
        half_batch_size = batch_size / 2
        product_size = half_batch_size * half_batch_size
        trip_data = np.ones((product_size, 3), dtype=np.int)
        for epoch in range(self.num_epochs):
            # shuffle trip array, equals to shuffle the data, but can keep the relation between data and hard trip list
            for batch_id in range(self.data_size):
                pos_inds = np.random.permutation(self.pairs['pos'][batch_id])[:half_batch_size]
                # neg_inds = np.random.choice(self.pairs[batch_id]['neg'], batch_size-num_pos_inds)
                neg_class_ids = np.random.choice(np.arange(self.data_size), batch_size - half_batch_size)
                neg_inds = [np.random.choice(self.pairs['pos'][neg_class_id]) for neg_class_id in neg_class_ids]
                batch_inds = list(itertools.chain(pos_inds, neg_inds))

                trip_data[:, 0] = self.pairs['p2p'][0]
                trip_data[:, 1] = self.pairs['p2p'][1]
                trip_data[:, 2] = self.pairs['p2n'][1]

                indices = np.random.permutation(np.arange(product_size))

                data_batch = self.data[batch_inds, ::]
                label_batch = self.label[batch_inds]
                trip_data_batch = np.reshape(trip_data[indices, ::], [batch_size, -1, 3])

                yield zip(data_batch, label_batch, trip_data_batch)

    def hard_trip_batch_iter(self):
        # generate a batch iterator with paired data
        batch_size = self.batch_size
        pos_pair_size = self.batch_p*self.batch_k*(self.batch_k-1)
        product_size = self.batch_p*self.batch_k*self.batch_k*(self.batch_p-1)
        trip_data = np.ones((product_size, 4), dtype=np.int)
        temp_trip_data = np.ones((batch_size, product_size/batch_size, 2), dtype=np.int)
        valid_dim = pos_pair_size/batch_size
        for epoch in range(self.num_epochs):
            for batch_id in range(self.data_size):
                batch_inds = get_batch_inds(self.pairs, self.batch_p, self.batch_k)

                # trip_data[:pos_pair_size, 0] = self.pairs['p2p'][:, 0]
                # trip_data[:pos_pair_size, 1] = self.pairs['p2p'][:, 1]

                temp_trip_data[:, :valid_dim, 0] = np.reshape(self.pairs['p2p'][:, 0], (batch_size, -1))
                temp_trip_data[:, :valid_dim, 1] = np.reshape(self.pairs['p2p'][:, 1], (batch_size, -1))
                trip_data[:, :2] = np.reshape(temp_trip_data, (-1, 2))
                trip_data[:, 2] = self.pairs['p2n'][:, 0]
                trip_data[:, 3] = self.pairs['p2n'][:, 1]

                # indices = np.random.permutation(np.arange(product_size))
                trip_indices = np.random.permutation(np.arange(batch_size))

                data_batch = self.data[batch_inds, ::]
                label_batch = self.label[batch_inds]
                # trip_data_batch = np.reshape(trip_data[indices, ::], [batch_size, -1, 4])
                trip_data_batch = np.reshape(trip_data, [batch_size, -1, 4])[trip_indices, ::]

                yield zip(data_batch, label_batch, trip_data_batch)


def get_dataset_dir(name, dataset_dir):
    if name in ['CUHK03_New_ZL_D', 'CUHK03_New_ZL_L']:
        dataset_dir = os.path.join(dataset_dir, 'CUHK03_New_ZL')
    # elif name in ['CUHK01_AB', 'VIPeR', 'PRID', '3DPeS', 'i-LIDS_p80', 'i-LIDS_p50', 'i-LIDS_p30', 'GRID']:
    #     dataset_dir = os.path.join(dataset_dir, '{}_TT'.format(name))
    elif name in ['CUHK01_AB', '3DPeS', 'i-LIDS_p80', 'i-LIDS_p50', 'i-LIDS_p30']:
        dataset_dir = os.path.join(dataset_dir, '{}_TT'.format(name))
    elif name in ['VIPeR_Data']:
        dataset_dir = os.path.join(dataset_dir, '{}'.format(name[:-5]))
    elif name in ['Viper']:
        dataset_dir = 'VIPeR'.join(dataset_dir.split('Viper'))
    elif name in ['i-LIDS']:
        dataset_dir = 'iLIDS'.join(dataset_dir.split('i-LIDS'))
    # pdb.set_trace()
    print("dataset_dir: %s" % dataset_dir)
    #assert os.path.isdir(dataset_dir)

    return dataset_dir


def gen_data_pairs(label_data, min_width, depth, pair_type):
    num_class = max(label_data) + 1
    unique_ids = np.sort(np.unique(label_data)).tolist()

    pair_info = {'pos': []}

    assert num_class == len(unique_ids), "the labels are not consistent"

    for index in xrange(num_class):
        pos_ids = np.where(label_data==index)[0].tolist()
        pos_width = len(pos_ids)
        if pos_width < min_width:
            comp_width = min_width - pos_width
            pos_ids.extend(np.array(pos_ids)[np.remainder(np.arange(comp_width), pos_width)].tolist())
        # pair_info['neg'] = [x for x in range_list if x not in pair_info['pos']] # not apply to hard triplet pair
        pair_info['pos'].append(pos_ids)

    if pair_type in ['pair', 'trip']:

        pos_rel_inds = range(min_width)
        neg_rel_inds = range(min_width, min_width+min_width)
        pos_to_pos_pairs = zip(*list(itertools.product(pos_rel_inds, pos_rel_inds)))
        pos_to_neg_pairs = zip(*list(itertools.product(pos_rel_inds, neg_rel_inds)))

        pair_info['p2p'] = pos_to_pos_pairs
        pair_info['p2n'] = pos_to_neg_pairs

    elif pair_type == 'hard':

        pos_to_pos_pairs = np.zeros((min_width, depth, depth-1, 2))
        pos_to_neg_pairs = np.zeros((min_width, depth, depth*(min_width-1), 2))
        all_indices = np.arange(min_width * depth).tolist()

        for ind_p in range(min_width):
            depth_offset = ind_p * depth
            pos_to_pos_pair_for_ind_p = zip(*list(itertools.permutations(range(depth_offset, depth_offset + depth), 2)))
            pos_to_pos_pairs[ind_p, :, :, 0] = np.reshape(pos_to_pos_pair_for_ind_p[0], (depth, -1))
            pos_to_pos_pairs[ind_p, :, :, 1] = np.reshape(pos_to_pos_pair_for_ind_p[1], (depth, -1))
            for ind_k in range(depth):
                pos_to_neg_pairs[ind_p, ind_k, :, 0] = depth_offset + ind_k
                pos_to_neg_pairs[ind_p, ind_k, :, 1] = all_indices[:depth_offset] + all_indices[(depth_offset + depth):]

        pair_info['p2p'] = np.reshape(pos_to_pos_pairs, (-1, 2))
        pair_info['p2n'] = np.reshape(pos_to_neg_pairs, (-1, 2))

    else:
        raise Exception('Pair type error')

    return pair_info, num_class


def get_batch_inds(pairs, batch_p, batch_k):
    num_class = len(pairs['pos'])
    batch_inds = []
    batch_ps = np.random.choice(num_class, batch_p, replace=False)
    # import pdb
    # pdb.set_trace()
    for ind_p in range(batch_p):
        batch_inds.extend(np.random.choice(pairs['pos'][batch_ps[ind_p]], batch_k, replace=True))
    return batch_inds
