import setuptools

with open('README.md', 'r') as infile:
    long_description = infile.read()

setuptools.setup(
    name='run_torch_model',
    version='0.1.4',
    author='Christer Dreierstad',
    author_email='christerdr@outlook.com',
    description='Package for training and evaluating neural network models made using pytorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/chdre/run-torch-model',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=['torch'],
    include_package_data=True,
)
