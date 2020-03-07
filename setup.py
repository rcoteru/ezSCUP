from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='ezSCUP',
      version='0.1.1',
      description='SCALE-UP Python Wrapper',
      long_description=long_description,
      url='https://github.com/rcote98/ezSCUP',
      author='Raúl Coterillo',
      author_email='raulcote98@gmail.com',
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
