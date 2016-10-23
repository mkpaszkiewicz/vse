import abc
import os

from vse.utils import *
from vse.error import *


__all__ = ['Index',
           'ForwardIndex',
           'InvertedIndex'
           ]


class Index:
    def __init__(self, dir_path, override=False, remove_on_exit=False):
        """All indexes base class. Removes existing index directory.
        Registers exit function, which will delete index directory on termination"""
        self._dir_path = complete_path(dir_path)
        if override:
            rmdir_if_exist(dir_path)
            os.makedirs(dir_path)
        if remove_on_exit:
            import atexit
            atexit.register(rmdir_if_exist, dir_path)

    @abc.abstractmethod
    def add(self, filename, hist, image):
        pass

    @abc.abstractmethod
    def get(self, query_hist):
        pass

    @abc.abstractmethod
    def remove(self, filename):
        pass


class ForwardIndex(Index):
    def __init__(self, dir_path='./index'):
        Index.__init__(self, dir_path)
        self._index = {}

    def add(self, filename, hist, image):
        if filename in self._index.keys():
            raise DuplicatedImageError(filename)
        self._index[filename] = hist

    def get(self, query_hist):
        return self._index.items()

    def remove(self, filename):
        if filename not in self._index.keys():
            raise NoImageError(filename)
        else:
            del self._index[filename]

    def __len__(self):
        return len(self._index)


class InvertedIndex(Index):
    def __init__(self, vw_amount, dir_path='./index', cutoff=2.0):
        Index.__init__(self, dir_path)
        self._index = [{} for i in range(vw_amount)]
        self._cutoff = cutoff

    def add(self, filename, hist, image):
        for i, vw in enumerate(hist):
            if vw > self._cutoff / len(hist):
                if filename in self._index[i].keys():
                    raise DuplicatedImageError(filename)
                self._index[i][filename] = hist

    def get(self, query_hist):
        results = []
        for i, vw in enumerate(query_hist):
            if vw > self._cutoff / len(query_hist):
                results.extend(self._index[i].items())
        return dict(results).items()

    def remove(self, filename):
        found = False
        for i, vw_dict in enumerate(self._index):
            if filename in vw_dict.keys():
                found = True
                del self._index[i][filename]
        if not found:
            raise NoImageError(filename)

    def __len__(self):
        return len(set(key for dic in self._index for key in dic.keys()))
