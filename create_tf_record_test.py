import os
import tensorflow as tf
from create_tf_record import create_sample


class CreateTFRecordTest(tf.test.TestCase):
    def _assertProtoEqual(self, proto_field, expectation):
        proto_list = [p for p in proto_field]
        self.assertListEqual(proto_list, expectation)

    def test_dict_to_tf_example(self):
        image_file = '61.jpg'
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
        example = create_sample(image_file, data_dir)

        self._assertProtoEqual(
            example.features.feature['image/height'].int64_list.value, [340])
        self._assertProtoEqual(
            example.features.feature['image/width'].int64_list.value, [453])
        self._assertProtoEqual(
            example.features.feature['image/filename'].bytes_list.value,
            [image_file])
        self._assertProtoEqual(
            example.features.feature['image/source_id'].bytes_list.value,
            [image_file])
        self._assertProtoEqual(
            example.features.feature['image/format'].bytes_list.value, ['jpeg'])
        self._assertProtoEqual(
            example.features.feature['image/object/bbox/xmin'].float_list.value,
            [90.0 / 453])
        self._assertProtoEqual(
            example.features.feature['image/object/bbox/ymin'].float_list.value,
            [104.0/340])
        self._assertProtoEqual(
            example.features.feature['image/object/bbox/xmax'].float_list.value,
            [1.0])
        self._assertProtoEqual(
            example.features.feature['image/object/bbox/ymax'].float_list.value,
            [336.0/340])
        self._assertProtoEqual(
            example.features.feature['image/object/class/text'].bytes_list.value,
            ['panda'])
        self._assertProtoEqual(
            example.features.feature['image/object/class/label'].int64_list.value,
            [1])


if __name__ == '__main__':
    tf.test.main()
