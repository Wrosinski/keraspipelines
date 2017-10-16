from setuptools import find_packages, setup

setup(name='Keras Pipelines',
      version='0.1',
      description='Pipelines for Keras',
      author='Wojciech Rosinski',
      author_email='wojtusior@gmail.com',
      url='https://github.com/fchollet/keras-pipelines',
      license='MIT',
      install_requires=['numpy>=1.9.1',
                        'pandas>=0.20.0',
                        'tensorflow>=1.3.0',
                        'keras>=2.0.8',
                        'scikit-learn>=0.19.0'],
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      keywords='deeplearning machinelearning keras tensorflow pipeline',
      packages=find_packages())
