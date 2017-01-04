from setuptools import setup

setup(name='vse',
      version='0.1.5',
      author='Marcin K. Paszkiewicz',
      author_email='mkpaszkiewicz@gmail.com',
      description='A visual search engine using local features descriptors and bag of words, based on OpenCV',
      url='https://github.com/mkpaszkiewicz/vse',
      download_url='https://github.com/mkpaszkiewicz/vse/tarball/0.1.5',
      packages=['vse'],
      keywords=['visual search engine computer vision local descriptors BoW'],
      install_requires=[
          'NumPy'
      ]
      )
