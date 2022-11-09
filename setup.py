from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Education',
    'Operating System :: Linux :: Ubuntu :: Kerner 5.15.0-52-generic',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
]

setup(
    name='cheMLearning',
    version='0.0.1',
    description='Tools for Metabolomics Multivariate Analysis',
    long_description=open('README.txt').read() + '\n\n' +
    open('CHANGELOG.txt').read(),
    url='',
    author='Christian Peralta',
    author_email='christian_94_97@hotmail.com',
    license='MIT',
    classifiers=classifiers,
    keywords='calculator',
    packages=find_packages(),
    install_requires=['numpy', 'pandas',
                      'scikit-learn', 'plotly', 'scipy', 'xlrd']
)
