from setuptools import setup, find_packages

classifiers = [
    'Intended Audience :: Education',
    'Operating System :: Linux :: Ubuntu :: Kerner 5.15.0-52-generic',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
]

with open("README.md", "r") as fh:
    long_desciption = fh.read()

setup(
    name='cheMLearning',
    version='0.0.1',
    description='Tools for Metabolomics Multivariate Analysis',
    long_description=long_desciption,
    url='',
    author='Christian Peralta',
    author_email='christian_94_97@hotmail.com',
    license='MIT',
    classifiers=classifiers,
    keywords='metabolomics',
    packages=find_packages(),
    install_requires=['numpy >= 1.23.4',
                      'pandas >= 1.5.1',
                      'plotly >= 5.11.0',
                      'scikit-learn >= 1.1.3',
                      'scipy >= 1.9.3',
                      'xlrd >= 2.0.1']
)
