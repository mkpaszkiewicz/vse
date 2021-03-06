import unittest
from unittest.mock import Mock, patch, mock_open

from vse import *


class UtilityTest(unittest.TestCase):
    @patch('vse.utils.convert_image')
    @patch('vse.utils.cv2.imread')
    def test_should_load_converted_image(self, mock_cv_imread, mock_convert_image):
        converted_image = Mock()
        mock_convert_image.return_value = converted_image

        img = load_image('dir/filename/')

        mock_cv_imread.assert_called_with('dir/filename/', cv2.IMREAD_GRAYSCALE)
        self.assertEqual(img, converted_image)

    @patch('vse.utils.cv2.imread')
    def test_should_raise_exception_if_failed_to_load_image(self, mock_cv_imread):
        mock_cv_imread.return_value = None
        self.assertRaises(ImageLoaderError, load_image, 'dir/filename/')

    @patch('vse.utils.convert_image')
    @patch('vse.utils.cv2.imdecode')
    def test_should_load_image_from_buffer(self, mock_cv_imdecode, mock_convert_image):
        converted_image = Mock()
        mock_convert_image.return_value = converted_image
        buffer = numpy.fromstring('datadata')

        img = load_image_from_buf(buffer)

        self.assertTrue(mock_cv_imdecode.called)
        self.assertTrue(mock_cv_imdecode.call_args, (numpy.frombuffer(buffer, numpy.uint8), cv2.IMREAD_GRAYSCALE))
        self.assertEqual(img, converted_image)

    def test_should_not_load_image_from_empty_buffer(self):
        self.assertRaises(ImageLoaderError, load_image_from_buf, numpy.fromstring(''))

    @patch('vse.utils.cv2.imdecode')
    def test_should_raise_exception_if_failed_to_load_image_from_buffer(self, mock_cv_imdecode):
        buffer = numpy.fromstring('datadata')
        mock_cv_imdecode.return_value = None

        self.assertRaises(ImageLoaderError, load_image_from_buf, buffer)

    def test_should_raise_error_if_image_too_small(self):
        image_mock = Mock()
        image_mock.shape = [IMAGE_MIN_SIZE / 2, IMAGE_MIN_SIZE / 2]

        self.assertRaises(ImageSizeError, convert_image, image_mock)

    def test_should_not_resize_image_with_proper_size(self):
        image_mock = Mock()
        image_mock.shape = [IMAGE_MAX_SIZE / 2, IMAGE_MAX_SIZE / 2]

        converted_img = convert_image(image_mock)

        self.assertEqual(image_mock, converted_img)

    @patch('vse.utils.cv2.resize')
    def test_should_resize_image(self, mock_cv_resize):
        image_mock = Mock()
        image_mock.shape = [IMAGE_MAX_SIZE * 2, IMAGE_MAX_SIZE * 3]
        converted_image_mock = Mock()
        mock_cv_resize.return_value = converted_image_mock

        converted_img = convert_image(image_mock)

        self.assertTrue(mock_cv_resize, converted_img)
        mock_cv_resize.assert_called_with(image_mock, (int(IMAGE_MAX_SIZE), int(2 / 3 * IMAGE_MAX_SIZE)), interpolation=cv2.INTER_AREA)
        self.assertEqual(converted_img, converted_image_mock)

    @patch('vse.utils.load_image')
    def test_should_get_images_generator(self, load_image_mock):
        paths = ['dir/filename1/', 'dir/filename2/']
        for i, image in enumerate(load_images(paths)):
            load_image_mock.assert_called_with(paths[i])

    @patch('vse.utils.shutil')
    def test_should_remove_directory(self, shutil_mock):
        rmdir_if_exist('dir/filename/')
        shutil_mock.rmtree.assert_called_with('dir/filename/')

    @patch('vse.utils.shutil')
    def test_should_not_remove_non_existing_directory(self, shutil_mock):
        exc = OSError()
        exc.errno = errno.ENOENT
        shutil_mock.rmtree.side_effect = exc

        rmdir_if_exist('dir/filename/')
        shutil_mock.rmtree.assert_called_with('dir/filename/')

    @patch('vse.utils.shutil')
    def test_should_fail_removing_broken_directory(self, shutil_mock):
        shutil_mock.rmtree.side_effect = OSError
        self.assertRaises(OSError, rmdir_if_exist, 'dir/filename/')
        shutil_mock.rmtree.assert_called_with('dir/filename/')

    def test_should_complete_dir_path(self):
        path = complete_path('./example/path')
        self.assertEqual(path, './example/path/')

    def test_should_not_change_dir_path(self):
        path = complete_path('./example/path/')
        self.assertEqual(path, './example/path/')

    @patch('vse.utils.pickle')
    def test_should_load_data_from_file(self, pickle_mock):
        with patch('builtins.open', mock_open()) as file_mock:
            load('dir/filename/')
            file_mock.assert_called_with('dir/filename/', 'rb')
            pickle_mock.load.assert_called_with(file_mock())

    @patch('vse.utils.pickle')
    def test_should_save_data_to_file_using_highest_protocol(self, pickle_mock):
        data = 'data'
        with patch('builtins.open', mock_open(read_data=data)) as file_mock:
            save('dir/filename/', data)
            file_mock.assert_called_with('dir/filename/', 'wb')
            pickle_mock.dump.assert_called_with(data, file_mock(), pickle.HIGHEST_PROTOCOL)

    @patch('vse.utils.pickle')
    def test_should_save_data_to_file_using_default_protocol(self, pickle_mock):
        data = 'data'
        with patch('builtins.open', mock_open(read_data=data)) as file_mock:
            save('dir/filename/', data, pickle.DEFAULT_PROTOCOL)
            file_mock.assert_called_with('dir/filename/', 'wb')
            pickle_mock.dump.assert_called_with(data, file_mock(), pickle.DEFAULT_PROTOCOL)
