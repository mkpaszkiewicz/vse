from setuptools import setup

setup(name='vse',
      version='0.1',
      author='Marcin K. Paszkiewicz',
      author_email='mkpaszkiewicz@gmail.com',
      description='Configurable visual search engine based on the OpenCV',
      url='https://github.com/mkpaszkiewicz/vse',
      packages=['vse'],
      keywords=['visual', 'search', 'engine', 'computer', 'vision'],
      install_requires=[
          'NumPy',
          'scipy',
      ]
      )
