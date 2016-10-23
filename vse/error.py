import vse.engine


class VisualSearchEngineError(Exception):
    """Base class for vse exceptions."""

    def __init__(self, msg=''):
        self.message = msg
        Exception.__init__(self, msg)

    def __repr__(self):
        return self.message

    __str__ = __repr__


class DuplicatedImageError(VisualSearchEngineError):
    """Raised if added image already exist in image index"""

    def __init__(self, image_path):
        msg = 'Image \'' + image_path + '\' already exists in the index'
        VisualSearchEngineError.__init__(self, msg)


class NoImageError(VisualSearchEngineError):
    """Raised if deleted non-existent image in image index"""

    def __init__(self, image_path):
        msg = 'Image \'' + image_path + '\' does not exist in the index'
        VisualSearchEngineError.__init__(self, msg)


class ImageSizeError(VisualSearchEngineError):
    """Raised if loaded image width or height is smaller than IMAGE_MIN_SIZE"""

    def __init__(self, image_path=''):
        if image_path:
            msg = 'Both width and height of the \'' + image_path + '\' must be greater than ' + str(vse.engine.IMAGE_MIN_SIZE)
        else:
            msg = 'Both width and height of the image must be greater than ' + str(vse.engine.IMAGE_MIN_SIZE)
        VisualSearchEngineError.__init__(self, msg)


class ImageLoaderError(VisualSearchEngineError):
    """Raised if cannot read image from file or buffer"""

    def __init__(self, image_path=''):
        if image_path:
            msg = 'Cannot read file: \'' + image_path + '\''
        else:
            msg = 'Cannot read image from buffer'
        VisualSearchEngineError.__init__(self, msg)
