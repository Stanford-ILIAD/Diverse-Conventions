#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='comedi',
      version='0.0.1',
      description='CoMeDi',
      author='',
      author_email='',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          'torch',
          'numpy',
          'tqdm',
          'gym',
          'hanabi_learning_environment',
          "tensorboard",
          "onnx",
          "onnxruntime",
          "onnx-tf",
          "tensorflow",
          "tensorflow-addons",
          "tensorflowjs",
          "simple-onnx-processing-tools",
          "nvidia-pyindex",
          "tensorflow-probability",
          "pyarrow",
          "flask"
      ],
      )
