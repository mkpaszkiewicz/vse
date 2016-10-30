import cv2
import numpy
from abc import ABCMeta, abstractmethod


__all__ = ['HistComparator',
           'Correlation',
           'ChiSquared',
           'Intersection',
           'Hellinger',
           'Bhattacharyya',
           'ChiSquaredAlt',
           'KullbackLeibler',
           'Euclidean',
           'CosineAngle',
           ]


class HistComparator(metaclass=ABCMeta):
    REVERSED = False

    @abstractmethod
    def compare(self, h1, h2):
        """"Compares histograms. Returns comparison metric"""
        pass


class Correlation(HistComparator):
    REVERSED = True

    def compare(self, h1, h2):
        return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)


class ChiSquared(HistComparator):
    def compare(self, h1, h2):
        return cv2.compareHist(h1, h2, cv2.HISTCMP_CHISQR)


class Intersection(HistComparator):
    REVERSED = True

    def compare(self, h1, h2):
        return cv2.compareHist(h1, h2, cv2.HISTCMP_INTERSECT)


class Hellinger(HistComparator):
    def compare(self, h1, h2):
        return cv2.compareHist(h1, h2, cv2.HISTCMP_HELLINGER)


class Bhattacharyya(HistComparator):
    def compare(self, h1, h2):
        return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)


class ChiSquaredAlt(HistComparator):
    def compare(self, h1, h2):
        return cv2.compareHist(h1, h2, cv2.HISTCMP_CHISQR_ALT)


class KullbackLeibler(HistComparator):
    def compare(self, h1, h2):
        return cv2.compareHist(h1, h2, cv2.HISTCMP_KL_DIV)


class Euclidean(HistComparator):
    def compare(self, h1, h2):
        return numpy.linalg.norm(h1 - h2)


class CosineAngle(HistComparator):
    REVERSED = True

    def compare(self, h1, h2):
        return cosine_angle(h1, h2)


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / numpy.linalg.norm(vector)


def cosine_angle(v1, v2):
    """Returns the cosine angle between vectors v1 and v2."""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return numpy.dot(v1_u, v2_u)
