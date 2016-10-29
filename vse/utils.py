import errno
import shutil

import cv2
import numpy
import pickle

from vse.error import *

IMAGE_MAX_SIZE = 1000
IMAGE_MIN_SIZE = 150


def load_image(filename):
    """Reads an image from file. Image is being converted to grayscale and resized."""
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ImageLoaderError(filename)
    return convert_image(image)


def load_image_from_buf(buf):
    """Reads an image from a buffer in memory. Image is being converted to grayscale and resized."""
    if len(buf) == 0:
        raise ImageLoaderError()
    image = cv2.imdecode(numpy.frombuffer(buf, numpy.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ImageLoaderError()
    return convert_image(image)


def convert_image(image, filename=''):
    """Image is being resized according to IMAGE_MAX_SIZE.
    Raises ImageLoaderException if image height or width is smaller than IMAGE_MIN_SIZE.
    """
    height, width = image.shape[:2]
    if max(width, height) > IMAGE_MAX_SIZE:
        scale = IMAGE_MAX_SIZE / max(height, width)
        image = cv2.resize(image, (int(scale * width), int(scale * height)), interpolation=cv2.INTER_AREA)
    elif min(width, height) < IMAGE_MIN_SIZE:
        raise ImageSizeError(filename)
    return image


def load_images(filenames):
    """Image generator."""
    for filename in filenames:
        yield load_image(filename)


def rmdir_if_exist(dir_path):
    """Removes directory if exists. Raises exception if other than ENOENT."""
    try:
        shutil.rmtree(dir_path)
    except OSError as exception:
        if exception.errno != errno.ENOENT:
            raise


def complete_path(path):
    """Adds '/' at the end of path if not exist."""
    if path[-1] != '/':
        path += '/'
    return path


def load(filename):
    """Loads data from file using pickle."""
    with open(filename, 'rb') as file:
        return pickle.load(file)


def save(filename, data, protocol=pickle.HIGHEST_PROTOCOL):
    """Saves data to file using pickle."""
    with open(filename, 'wb') as file:
        pickle.dump(data, file, protocol)


def normalize(hist):
    """Normalizes histogram by casting values to [0, 1]."""
    return numpy.array([val / sum(hist) for val in hist], dtype=numpy.float32)
