from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="eve",
    version="0.1",
    author="EVE2 Team",
    author_email="your.email@example.com",
    description="Interactive robotic system inspired by EVE from Wall-E",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/eve2",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        'opencv-python',
        'numpy',
        'face_recognition',
    ],
    entry_points={
        "console_scripts": [
            "eve2=scripts.start_eve:main",
        ],
    },
) 