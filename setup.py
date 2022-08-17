# This file is adapted from https://git.corp.adobe.com/3di/python-scaffold
from setuptools import find_packages, setup

# This file is needed so you can install your package via pip.
setup(
    name="multi_meta_ssd",
    packages=find_packages(include=["multi_meta_ssd", "multi_meta_ssd.*"]),
    version="0.1-dev",
    author="Meryem M'hamdi",
    author_email="mmhamdi@adobe.com",
    # If you actually want to be able to install this via pip,
    # you have to put your python dependencies here as well. For now,
    # since we only use this file to install in dev mode (setup.sh and setup.ps1)
    # this is not needed.
    description="Example of creating a python repository / package",
    long_description_content_type="README.md",
    url="https://git.corp.adobe.com/euclid/python-project-scaffold",
    python_requires=">=3.7",
    # All the packages that your code depends on to run
    # Note that development dependencies should go into the conda yaml file.
    # For deps like boa_toolkit that reside in private repos, you still need
    # to install them manually in conda.yaml, putting them here, just checks
    # that the package is installed already
    install_requires=[
        # Libraries that users of your lib also need should go here.
        # Shared portion of python project scaffold.
        # The reason that this code is not directly included as a part of python-project-scaffold is that
        # when we update that code, you as a user of python-project-scaffold can just bump the version of
        # boa_toolkit and get the updates. Otherwise, the updates should be copied over individually.
        "boa_toolkit @ git+ssh://git@git.corp.adobe.com/euclid/boa_toolkit@v2.0.4",
    ],
    # enables running from the terminal using just the entry point name and the args
    # We add a single entry point multi_meta_ssd, from which you can execute all your actions.
    entry_points={
        "console_scripts": [
            "multi_meta_ssd=multi_meta_ssd.commands.main:main_cmd",
        ]
    },
)
