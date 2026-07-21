from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name="omnia-sdk",
    version="0.1.0",
    description=".omnia — Medical Image Container for AI Training. One file instead of 277.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mishel Adnan",
    author_email="mishel@example.com",
    url="https://github.com/mishel-0/omnia-sdk",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "omnia=omnia_sdk.cli:main",
        ],
    },
    python_requires=">=3.9",
    install_requires=[
        "pydicom>=3.0",
        "zstd>=1.5",
        "numpy>=1.24",
    ],
    extras_require={
        "ml": ["torch>=2.0", "torchvision>=0.15"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: System :: Archiving :: Compression",
    ],
)
