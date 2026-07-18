from setuptools import setup, find_packages

setup(
    name="omnia-sdk",
    version="2.0.0",
    description="Omnia DICOM Compression SDK — 3x lossless, 277 files → 1",
    author="Omnia AI",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pydicom>=2.0",
        "numpy>=1.20",
        "imagecodecs>=2023",
        "zstd>=1.5",
    ],
    entry_points={
        "console_scripts": [
            "omnia=omnia.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
    ],
)
