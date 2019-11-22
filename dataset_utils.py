import tensorflow as tf

from PIL import Image
import numpy as np
import skimage.io as io

import scipy.misc
# import matplotlib.pyplot as plt # some error here

import os
import pdb

"""
TFRecord tutorial style generation

class ImageReader:  retrieval the spatial info from string encoded object (required by .tfrecord files);
                    it is provied us a way to decode string encoded object back to image arrays;
                    the internal decoding function (graph) need a tf session to run it.
                    Still not sure how excatly this can be done???

decode the image string data to image array
decode_img_string_tf(img_string): no need external height and width info,
but the decode graph (as a ImageReader object) and a session is needed to run it.
"""


class ImageReader(object):
    """Helper class that provides TF image coding utilities.
    since input image as a string, so the height and width is not direct
    this class is focus on retrieval spatial indormation from string data.
    """

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        # here a graph is defined?
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    # here provides the decode methods!!!!! from string encoded to array data
    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        # image_data is string
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def decode_img_string_tf(img_string, decoder, sess):
    # external run session,
    # external graph context,
    # graph in decoder.decode_jpeg.

    image = decoder.decode_jpeg(sess, img_string)
    # pdb.set_trace()
    return image


"""
Alternative to TFRecord tutorial style generation:

Mainly based on Daniil's BLOG: Tfrecords Guide

image2string:
read the image file directly as array, the spatial info cam be get directly,
no need complicated methods (define graph and session run in
height, width = image_reader.read_image_dims(sess, image_data))
array to string also have simple API

decode the image string data to image array
decode_img_string_np: parsing the .tfrecord files in alternative way (no graph and session are needed)
"""


def image2string(img_file_name):
    img = np.array(Image.open(img_file_name))
    height = img.shape[0]
    width = img.shape[1]
    image_data = img.tostring()
    return image_data, height, width


def decode_img_string_np(img_string, height, width):
    # recover as a 1d array
    # dtype=np.uint8 is important
    img_1d = np.fromstring(img_string, dtype=np.uint8)
    # reshape to image
    return img_1d.reshape((height, width, -1))


"""
Transform the bunch of data to tfexample style
"""


def int64_feature(values):
    """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    a TF-Feature.
  """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    a TF-Feature.
  """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def ReID_image_to_tfexample(image_data, image_file_name, image_format, height, width,
                            PID, PL, CID, CP, LCV, LPI, DID, ImUID):
    # tf.train.Example is a very important data utility for
    # generating and parsing .tfrecord file
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/filename': bytes_feature(image_file_name),
        'image/format': bytes_feature(image_format),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/PID': int64_feature(PID),
        'image/PL': int64_feature(PL),
        'image/CID': int64_feature(CID),
        'image/CP': int64_feature(CP),
        'image/LCV': int64_feature(LCV),
        'image/LPI': int64_feature(LPI),
        'image/DID': int64_feature(DID),
        'image/ImUID': int64_feature(ImUID),
    }))

def Market_ReID_image_to_tfexample(image_data, image_file_name, image_format, height, width,
                            PID, PL, CID, Seq, frame, BB, DID, ImUID):
    # tf.train.Example is a very important data utility for
    # generating and parsing .tfrecord file
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/filename': bytes_feature(image_file_name),
        'image/format': bytes_feature(image_format),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/PID': int64_feature(PID),
        'image/PL': int64_feature(PL),
        'image/CID': int64_feature(CID),
        'image/Seq': int64_feature(Seq),
        'image/frame': int64_feature(frame),
        'image/BB': int64_feature(BB),
        'image/DID': int64_feature(DID),
        'image/ImUID': int64_feature(ImUID),
    }))


def SmallSet_ReID_image_to_tfexample(image_data, img_full_path, image_format, height, width,
                                PID, PL, CID, DID, ImUID):
    # tf.train.Example is a very important data utility for
    # generating and parsing .tfrecord file
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/filename': bytes_feature(img_full_path),
        'image/format': bytes_feature(image_format),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/PID': int64_feature(PID),
        'image/PL': int64_feature(PL),
        'image/CID': int64_feature(CID),
        'image/DID': int64_feature(DID),
        'image/ImUID': int64_feature(ImUID),
    }))


def MergeSets_ReID_image_to_tfexample(image_data, img_full_path, image_format, height, width,
                                PID_str, PL, CID, DID, ImUID):
    # tf.train.Example is a very important data utility for
    # generating and parsing .tfrecord file
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/filename': bytes_feature(img_full_path),
        'image/format': bytes_feature(image_format),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/PID': bytes_feature(PID_str),
        'image/PL': int64_feature(PL),
        'image/CID': int64_feature(CID),
        'image/DID': int64_feature(DID),
        'image/ImUID': int64_feature(ImUID),
    }))

# def show_image(img_array):



def parsing_compare_tfrecord(tfrecord_files, img_folder, _STRING_ENCODE_STYLE):
    """
    recover the information in tfrecord_file and compare with the original images

    :param tfrecord_files: the list of .tfrecord files
    :param img_folder: original images folder
    :param _STRING_ENCODE_STYLE: different ways to decode the string encoded image data
    :return:
    """

    for tfrecord_file in tfrecord_files:
        # record_iterator for iterating example string objects in tfrecord file, one by one
        record_iterator = tf.python_io.tf_record_iterator(path=tfrecord_file)

        for example_str_obj in record_iterator:
            # define an example object for parsing example string object
            example = tf.train.Example()
            # parsing the string and load the data
            example.ParseFromString(example_str_obj)

            # read the data out from example using
            # example.features.feature['name'].[type: int64_list, bytes_list, float_list].value
            # it correspinding to the tf.train.Example(features=tf.train.Features(feature={'name': format_data})) structure
            # 'name' need to be corespondented.
            # returned a list

            height = int(example.features.feature['image/height'].int64_list.value[0])
            width = int(example.features.feature['image/width'].int64_list.value[0])
            PID = int(example.features.feature['image/PID'].int64_list.value[0])
            CID = int(example.features.feature['image/CID'].int64_list.value[0])
            CP = int(example.features.feature['image/CP'].int64_list.value[0])
            LCV = int(example.features.feature['image/LCV'].int64_list.value[0])
            LPI = int(example.features.feature['image/LPI'].int64_list.value[0])

            ori_file_name_str = example.features.feature['image/filename'].bytes_list.value[0]
            print 'Ori file name: {}'.format(ori_file_name_str)
            img_full_name = os.path.join(img_folder, ori_file_name_str)
            assert os.path.isfile(img_full_name)
            img_ori_array = np.array(Image.open(img_full_name))
            print img_ori_array.dtype
            assert img_ori_array.shape[0] == height
            assert img_ori_array.shape[1] == width

            print 'Height: {}, Width: {}, PID: {}, CID: {},' \
                  ' CP: {}, LCV: {}, LPI: {}'.format(height, width, PID, CID, CP, LCV, LPI)

            # Discuss later: string data encoding and decoding
            # different ways to decode the string data according to different encode ways
            img_string = example.features.feature['image/encoded'].bytes_list.value[0]

            if _STRING_ENCODE_STYLE is 'tf':
                with tf.Graph().as_default():
                    # ImageReader: provide string data decoding function
                    image_reader = ImageReader()

                    # sess for run the graph
                    # (using ImageReader object)
                    with tf.Session('') as sess:
                        # Even though the image recovered from tf style encoding string is not totally (erery pixel) the same as
                        # original image, perceptionally no much difference (looks the same).
                        img_recover_array = decode_img_string_tf(img_string, image_reader, sess)

            elif _STRING_ENCODE_STYLE is 'np':
                # totally the same as original images
                img_recover_array = decode_img_string_np(img_string, height, width)
            else:
                raise ValueError('No such string encode style: {}'.format(_STRING_ENCODE_STYLE))

            # spatial info comparison
            assert img_recover_array.shape == img_ori_array.shape
            assert img_ori_array.shape[0] == height
            assert img_ori_array.shape[1] == width

            # pdb.set_trace()

            # can not plot, do not know why?????
            # show_image(img_ori_array)
            # show_image(img_recover_array)

            abs_diff = np.abs(img_ori_array - img_recover_array)
            print np.sum(np.asarray(abs_diff == 1, dtype=np.int64))
            print np.sum(np.asarray(abs_diff == 2, dtype=np.int64))
            print np.sum(np.asarray(abs_diff == 255, dtype=np.int64))

            # SAVE AND COMPARE
            # Even though the image recovered from tf style encoding string is not totally (erery pixel) the same as
            # original image, perceptionally no much difference (looks the same).
            img_cmp_path = '/import/vision-datasets001/xc302/ReID_Datasets/TFRecords/img_cmp'
            scipy.misc.imsave('{}/ori.jpg'.format(img_cmp_path), img_ori_array)
            scipy.misc.imsave('{}/recover.jpg'.format(img_cmp_path), img_recover_array)


            # pdb.set_trace()
            if np.allclose(img_ori_array, img_recover_array):
                print 'recover image is close to original image'
            else:
                print 'recover image is NOT close to original image'

            raw_input()
