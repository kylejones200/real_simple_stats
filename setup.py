from setuptools import setup, find_packages

setup(
    name='real_simple_stats',
    version='1.0',
    description='Python version of the Real Simple Stats Handbook',
    author='Kyle Jones',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy'
    ],
    license='MIT', 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
)