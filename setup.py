from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='projecc',
      version='1.0.1',
      description='A small simple package for projecting orbital elements onto sky plane and vice versa',
      url='https://github.com/logan-pearce/projecc',
      author='Logan Pearce',
      author_email='loganpearce1@arizona.edu',
      license='MIT',
      packages=['projecc'],
      install_requires=['numpy','astropy'],
      package_data={},
      include_package_data=True,
      zip_safe=False)