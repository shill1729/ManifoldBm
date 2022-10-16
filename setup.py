from setuptools import setup

setup(
    name='ManifoldBm',
    version='1.0',
    packages=['ManifoldBm'],
    url='https://github.com/shill1729/ManifoldBm',
    license='MIT',
    install_requires=[
          'numpy',
          'sympy',
          "matplotlib"
    ],
    author='S. Hill',
    author_email='52792611+shill1729@users.noreply.github.com',
    description='Tools for Brownian motion on Manifolds'
)
