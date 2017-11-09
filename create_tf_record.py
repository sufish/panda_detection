import tensorflow as tf
import os
import io
import PIL.Image
import json
import random
from os.path import basename
from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('image_dir', '', 'panda images dir')
flags.DEFINE_string('output_dir', '', 'output dir')
FLAGS = flags.FLAGS


def create_sample(image_filename, data_dir):
    image_path = os.path.join(data_dir, image_filename)
    annotation_path = os.path.join(data_dir, 'annotations', os.path.splitext(image_filename)[0] + ".json")
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    with open(annotation_path) as fid:
        image_annotation = json.load(fid)
    width = image_annotation['image_w_h'][0]
    height = image_annotation['image_w_h'][1]
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []

    for obj in image_annotation['objects']:
        classes.append(1)
        classes_text.append('panda')
        box = obj['x_y_w_h']
        xmins.append(float(box[0]) / width)
        ymins.append(float(box[1]) / height)
        xmaxs.append(float(box[0] + box[2] - 1) / width)
        ymaxs.append(float(box[1] + box[3] - 1) / height)

    filename = image_annotation['filename'].encode('utf8')
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def create_tf_record(example_file_list, data_dir, output_file_path):
    writer = tf.python_io.TFRecordWriter(output_file_path)
    for filename in example_file_list:
        tf_example = create_sample(filename, data_dir)
        writer.write(tf_example.SerializeToString())
    writer.close()


def main(_):
    data_dir = FLAGS.image_dir
    output_dir = FLAGS.output_dir
    all_examples = []
    for f in os.listdir(data_dir):
        if f.endswith(".jpeg") or f.endswith(".jpg"):
            all_examples.append(basename(f))
    random.seed(42)
    random.shuffle(all_examples)
    num_examples = len(all_examples)
    num_train = int(0.7 * num_examples)
    train_examples = all_examples[:num_train]
    val_examples = all_examples[num_train:]
    create_tf_record(train_examples, data_dir, os.path.join(output_dir, 'train.record'))
    create_tf_record(val_examples, data_dir, os.path.join(output_dir, 'val.record'))
    print('%d training and %d validation examples.',
          len(train_examples), len(val_examples))


if __name__ == '__main__':
    tf.app.run()
