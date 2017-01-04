vse
---

#### Description

**vse** is a [Python](https://www.python.org) package containing configurable visual search engine based on [OpenCV](http://www.opencv.org).

#### Requirements
Package requires [OpenCV](http://www.opencv.org). To install it I highly recommend using dependency and environment manager - [Anaconda](https://www.continuum.io/).

Run below commands in terminal:
```
$ conda create -n opencv numpy python=3
$ source activate opencv
$ conda install -c https://conda.binstar.org/menpo opencv3
```

```conda create``` makes a new virtual environment where ```-n``` flag set the environment name.
The names after that are the packages that get installed upon creating the environment and the keyword argument ```python=3``` makes sure that we use python 3.
OpenCV is not included in Anaconda’s regular list of packages, so you need to use binstar to get it.

#### Installation
After all requirements are satisfied you can install **vse** by running:

```
$ pip install vse
```

### Author

Marcin K. Paszkiewicz <mkpaszkiewicz@gmail.com>  
*Warsaw University of Technology*
