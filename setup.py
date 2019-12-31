import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="diagram2vec", # Replace with your own username
    version="0.0.1",
    author="Oleg Kachan",
    author_email="oleg.n.kachan@gmail.com",
    description="Vector Representations of Persistence Diagrams",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oleg-kachan/diagram2vec",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)