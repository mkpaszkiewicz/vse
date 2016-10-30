import abc
from vse.error import *


class Index:
    def __init__(self, ranker):
        self.ranker = ranker
        self.vw_freq = []

    @abc.abstractmethod
    def find(self, query_hist, n):
        pass

    def __setitem__(self, image_id, hist):
        self._add(image_id, hist)
        self._update_freq_after_addition(hist)

    @abc.abstractmethod
    def _add(self, image_id, hist):
        pass

    def _update_freq_after_addition(self, hist):
        if not self.vw_freq:
            self.vw_freq = hist.copy()
        self.vw_freq = [(n_freq * (len(self) - 1) + n) / len(self) for n, n_freq in zip(hist, self.vw_freq)]

    def __delitem__(self, image_id):
        hist = self[image_id]
        self._remove(image_id)
        self._update_freq_after_deletion(hist)

    @abc.abstractmethod
    def _remove(self, image_id):
        pass

    def _update_freq_after_deletion(self, hist):
        if len(self) == 0:
            self.vw_freq = []
        self.vw_freq = [(n_freq * (len(self) + 1) - n) / len(self) for n, n_freq in zip(hist, self.vw_freq)]

    @abc.abstractmethod
    def __getitem__(self, image_id):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass


class ForwardIndex(Index):
    def __init__(self, ranker):
        Index.__init__(self, ranker)
        self.index = {}

    def find(self, query_hist, n):
        return self.ranker.rank(query_hist, self.index.items(), n, self.vw_freq)

    def _add(self, image_id, hist):
        if image_id in self.index:
            raise DuplicatedImageError(image_id)
        self.index[image_id] = hist

    def _remove(self, image_id):
        if image_id not in self.index:
            raise NoImageError(image_id)
        del self.index[image_id]

    def __getitem__(self, image_id):
        return self.index[image_id]

    def __len__(self):
        return len(self.index)


class InvertedIndex(Index):
    def __init__(self, ranker, recognized_visual_words, cutoff=2.0):
        Index.__init__(self, ranker)
        self.index = [{} for i in range(recognized_visual_words)]
        self.cutoff = cutoff / recognized_visual_words

    def find(self, query_hist, n):
        return self.ranker.rank(query_hist, self._items(query_hist), n, self.vw_freq)

    def _items(self, query_hist):
        results = {}
        for visual_word_freq, subindex in zip(query_hist, self.index):
            if visual_word_freq > self.cutoff:
                results.update(subindex)
        return results.items()

    def _add(self, image_id, hist):
        for visual_word_freq, subindex in zip(hist, self.index):
            if visual_word_freq > self.cutoff:
                if image_id in subindex:
                    raise DuplicatedImageError(image_id)
                subindex[image_id] = hist

    def _remove(self, image_id):
        found = False
        for subindex in self.index:
            if image_id in subindex:
                found = True
                del subindex[image_id]
        if not found:
            raise NoImageError(image_id)

    def __getitem__(self, image_id):
        for subindex in self.index:
            if image_id in subindex:
                return subindex[image_id]
        raise KeyError(image_id)

    def __len__(self):
        return len(set(image_id for subindex in self.index for image_id in subindex))
