from setuptools import setup

setup(name='vse',
      version='1.0',
      author='Marcin K. Paszkiewicz',
      author_email='mkpaszkiewicz@gmail.com',
      description='Configurable visual search engine based on the OpenCV',
      packages=['vse', 'tests'],
      install_requires=[
          'NumPy',
          'scipy',
      ]
      )
