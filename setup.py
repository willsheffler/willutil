from setuptools import setup, find_packages

setup(
    name='willutil',
    version='0.1',
    url='https://github.com/willsheffler/willutil',
    author='Will Sheffler',
    author_email='willsheffler@gmail.com',
    description='Common utility stuff extracted from various projects',
    packages=find_packages(),
    install_requires=[
        'pytest',
        'tqdm',
    ],
)
