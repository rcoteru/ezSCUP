from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='ezSCUP',
      version='v1.0',
      description='SCALE-UP Python Wrapper',
      url='https://github.com/rcote98/ezSCUP',
      author='Raúl Coterillo',
      author_email='raulcote98@gmail.com',
      download_url = 'https://github.com/rcote98/ezSCUP/archive/v1.0.tar.gz',
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        ],
      install_requires=[
          "numpy",
          "pandas",
          "matplotlib"
      ],
      packages=find_packages(),
      zip_safe=False,
      python_requires='>=3.6'
      
    )
