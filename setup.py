#!/usr/bin/env python

from setuptools import setup

setup(name='pyvan',
      version='0.9',
      description='For assessment and classification of potentially variable stellar light-curves',
      author='Kellen D Lawson',
      author_email='kellenlawson@gmail.com',
      url='https://github.com/kdlawson/pyvan',
      packages=['pyvan'],
      package_dir={'pyvan': 'pyvan'},
      package_data={'pyvan': ['rrlyr_templates/*.dat']},
      install_requires=['lmfit','matplotlib','numpy','multiprocessing','scipy','joblib']
      )