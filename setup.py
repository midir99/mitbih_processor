import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="mitbih_processor",
    version="1.0.1",
    description="Process MIT-BIH Arrhythmia Database records with PyWavelets",
    long_description=README,
    url="https://github.com/jorge4larcon/mitbih_processor",
    author="Jorge Alarcon",
    author_email="jorge4larcon@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=["mitbih_processor", "webclient"],
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "numpy",
        "pandas",
        "PyWavelets",
        "scipy",
        "wfdb",
        "scikit-learn",
        "flask",
    ],
    entry_points={
        "console_scripts": [
            "mitbih_processor=mitbih_processor.__main__:main",
        ]
    },
)
