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
    """Raised when trying to add already existing image to the image index."""

    def __init__(self, image_path):
        msg = 'Image {} already exists in the index'.format(image_path)
        VisualSearchEngineError.__init__(self, msg)


class NoImageError(VisualSearchEngineError):
    """Raised when trying to delete non-existing image path from the image index."""

    def __init__(self, image_path):
        msg = 'Image {} does not exist in the index'.format(image_path)
        VisualSearchEngineError.__init__(self, msg)


class ImageSizeError(VisualSearchEngineError):
    """Raised if loaded image width or height is smaller than IMAGE_MIN_SIZE."""

    def __init__(self, image_path='image'):
        msg = 'Both width and height of the {} must be greater than {}'.format(image_path, vse.engine.IMAGE_MIN_SIZE)
        VisualSearchEngineError.__init__(self, msg)


class ImageLoaderError(VisualSearchEngineError):
    """Raised if cannot read image from file or buffer"""

    def __init__(self, image_path=''):
        if image_path:
            msg = 'Cannot read file: {}'.format(image_path)
        else:
            msg = 'Cannot read image from buffer'
        VisualSearchEngineError.__init__(self, msg)
