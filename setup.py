from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'ATLearn: An adaptive transfer learning toolkit'
LONG_DESCRIPTION = 'ATLearn is a Transfer Learning toolkit that supports easy model ' \
                   'building on top of the pre-trained models'

install_requires = [
    'Pillow==9.3.0',
    'matplotlib==3.5.2',
    'numpy==1.22.3',
    'opencv-python==4.5.5.64',
    'pandas==1.4.2',
    'scikit-learn==1.1.1',
    'tqdm==4.64.0',
    'torch==1.11.0',
    'torchvision==0.12.0',
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
