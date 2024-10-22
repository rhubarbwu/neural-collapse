# setup.py

from setuptools import find_packages, setup

setup(
    name="neural_collapse",
    version="0.1",
    author="Robert Wu",
    author_email="rupert@cs.toronto.edu",
    description="A generic library for accumulating feature statistics and computing neural collapse metrics.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rhubarbwu/neural-collapse",  # Link to your repository
    packages=find_packages(),  # Automatically find the 'matrix_operator' package
    install_requires=[
        "numpy",
        "scipy",
        "torch",
    ],
    extras_require={
        "faiss": ["faiss-gpu", "numpy<2"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
