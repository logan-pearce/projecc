from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='projecc',
      version='1.1.9',
      description='A small simple package for projecting orbital elements onto sky plane and vice versa',
      long_description='''projecc is a package for projecting Keplerian 2-body orbital elements into the sky plane
      and vice versa. It can be used to generate orbit tracks, survey completeness plots, and predicting the
      location of a planet in the sky plane over time. Find a tutorial here: https://github.com/logan-pearce/projecc/blob/main/projecc/UsingProjecc.ipynb. An interactive web app on predicting planet locations using projecc is here: https://reflected-light-planets.streamlit.app/. projecc is described in Pearce, Males, and Limbach (submitted to PASP)
        ''',
      url='https://github.com/logan-pearce/projecc',
      author='Logan Pearce',
      author_email='lapearce@umich.edu',
      license='MIT',
      packages=['projecc'],
      install_requires=['numpy','astropy'],
      package_data={},
      include_package_data=True,
      zip_safe=False)