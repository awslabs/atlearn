from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'ATLearn: An adaptive transfer learning toolkit'
LONG_DESCRIPTION = 'ATLearn is a Transfer Learning toolkit that supports easy model ' \
                   'building on top of the pre-trained models'

install_requires = [
    'Pillow==10.0.1',
    'matplotlib==3.8.0',
    'numpy==1.25.0',
    'opencv-python==4.7.0.72',
    'pandas==1.5.3',
    'scikit-learn==1.3.1',
    'tqdm==4.65.0',
    'torchvision==0.15.1',
    'tqdm==4.66.1',
    'torch==2.0.1',
    'torchvision==0.15.2',
]

# Setting up
setup(
    name="ATLearn",
    version=VERSION,
    author="Jun Wu",
    author_email="<junwu3@illinois.edu>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=install_requires,

    keywords=['python', 'first package'],
    classifiers=[
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ]
)
