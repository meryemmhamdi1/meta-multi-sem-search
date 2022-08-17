# Python Project Scaffold

A scaffold for creating your own cross-platform python project, and potentially your own python package installable by pip.

## What does it do? Why did we make it?

The answers to these questions can be found in the [wiki](https://git.corp.adobe.com/euclid/python-project-scaffold/wiki).

## Get started

Click the <img src="./docs/use-this-template.png" height=24> button to create a copy of this repo for your new project.  Give your new repo a name and a description and then click the  <img src="./docs/create-repository-from-template.png" height=24> button.  Github will create your repo and take you to its URL.  From there clone the repo to your computer so we can start customizing the template code for your new project.

## Make it your own!

In this scaffold we called the project `multi_meta_ssd`. **To use this scaffold for your own use case, you will want to rename it**. You can do this easily with the [tools/rename.py](tools/rename.py) script:

```
python tools/rename.py --snake-case awesome_lib --camel-case AwesomeLib --caps-case AWESOME_LIB
```

You can then delete the `tools/rename.py` script from your new repo.  You can also delete the `docs` folder, if you like.  It just contains images for this README.md.

If you want to, this is a good time to make your first git commit.

## Set up your environment

### Install Conda

You should have [conda](https://docs.conda.io/en/latest/) installed on your machine. If you don't you can use the the Conda package manager can be installed locally at any desired location on your system. 

- If on mac (also works for M1), run the [install-conda.sh](tools/install-conda.sh) script (no sudo needed)
    ```
    bash tools/install-conda.sh /path/to/install
    ```
- If on windows , run the [install-conda.ps1](tools/install-conda.ps1) script (no admin access needed)
    ```
    powershell tools/install-conda.ps1 \path\to\install
    ```

### Activating Conda

Here, I assume that you installed conda in `~/.miniconda3` for mac and `C:/miniconda3 `for windows.

After conda is installed, depending on what os and terminal you have, add these to that 
- Mac add to ~/.bash_profile and ~/.zshrc for bash and zsh, respectively.
  ```terminal
  source ~/.miniconda3/bin/activate
  ```
- windows powershell, add to $PROFILE (do `echo $PROFILE` in powershell to see where this file should be)
  ```terminal
  & C:\miniconda3\shell\condabin\conda-hook.ps1
  ```
- windows git bash, add to ~/.bash_profile
  ```terminal
  source /c/miniconda3/etc/profile.d/conda.sh
  ```

## Run the setup script

Based on your OS/Terminal combination, you need to add the following commands to initialize your terminal environment correctly. All the setup commands must be run from the repository root.  The first time you run this script it may take a while to install minconda, python, and any dependencies.  Subsequent runs of the script should be much faster.

### Windows + Powershell

Powershell (not CMD) is recommended on Windows. We recomend using [Windows Terminal](https://www.microsoft.com/en-us/p/windows-terminal/9n0dx20hk701) which provides a nice developer experience that supports powershell and several quality-of-life features like tabs, sensible copy-paste, Unicode support, etc.  

```
& tools\setup.ps1
```

### Linux/MacOS + bash

Note that `zsh` is the default on MacOS these days.  We don't guarantee this will work with `zsh` but it is [pretty easy to switch to `bash`](https://www.howtogeek.com/444596/how-to-change-the-default-shell-to-bash-in-macos-catalina/).

```
source tools/setup.sh
```

## Python installation and system PATH
* You do not need python installed separately, the setup command took care of it for you (enter `which python` in your terminal to confirm that you are now using the python installed in this project's virtual environment).  You should see `.venv` listed as the active virtual environment in your terminal now, just before the prompt.

```
(.venv) $ 
```

* You do not need to add `sys.path.append` at the beginning of your python scripts, the setup commands will make sure that your package (`multi_meta_ssd` in our case) is available to your python.

* Each time you create a new terminal and `cd` to the repo location, you will need to activate the virtual environment.

```
$ conda activate .venv/
```

* If you changed the dependencies in `tools/conda.yaml` or in `setup.py`, then you will need to `conda deactivate` and run the setup script again, in order to update your virtual environment.  

## Download Visual Studio Code (vscode)

Visual Studio Code (hereinafter referred to as __vscode__) offers a very nice development experience with auto complete, linting and debugger integration. It is highly recommeded to use vscode for development. You can download vscode from here: https://code.visualstudio.com/download

Once downloaded, open the vscode app and take care of a few one-time setup steps.

1. Click the <img src="./docs/extensions.png" height=24> button and type "python" into the search bar.  Install the "Python" extension. (first result)

   <img src="./docs/install-py-extension.png" width=250>

2. (Mac-only) Press Ctrl+Shift+P, type "shell", and select the "Shell Command: Install 'code' command in PATH" entry.
3. Close VS Code.  We'll reopen it later from the terminal, with the `code` command.

# Linting

We recommend using one of the standard Python linters, such as `pylint` or `flake8`.

# Unit testing and visual debugging

All the tests go into `tests` folder. Test input data should be kept under `data` which is saved in artifactory. Test temp output files will be created in the `.tmp_test_out` folder.

## Running the tests

You can run the tests with the `python tests/main.py` command.

If you like to run all the tests in the debugger, where you can add break points in vscode, open vscode, click on the little bug icon on the left, and click on the Play button.

<img src="./docs/run-all-tests.png" width=150>

To run a single test, open the test file in vscode, click on the lab flask icon on the left, then click on the rotating arrow.

<img src="./docs/erlen-flask.png" width=150>

Once you do this after a few seconds, on top of your test, there will be a `debug` text, click on that to run that specific test.

<img src="./docs/run-test.png" width=400>

## Adding a new test

Test should be inside `.py` files that start with the name `test_` and are inside the `tests` folder. They can also be in a subfolder of `tests`, but all subfolders must contain an `__init__.py` (even empty) file.

Here is an example of a simple test file:
```py
from multi_meta_ssd.multi_meta_ssd_test_case import MultiMetaSSDTestCase

# Tests are functions starting with `test_` of a class inheritting from `MultiMetaSSDTestCase`.
# All functions within the same class share the same setUp() and tearDown() functions.
class TestGroup1(MultiMetaSSDTestCase):
    def setUp(): #optional setup run once per every test functions
        pass

    def tearDown(): #optional tearDown run once per every test functions
        pass

    @classmethod
    def setUpClass(cls): #optional setup run once per test class
        pass

    @classmethod
    def tearDownClass(cls): #optional tearDown run once per test class
        pass

    def test_1(self): # test functions should start with
        sample_file = self.get_fixture_path("sample-file.txt")
        content = self.read_file(sample_file).strip()
        self.assertEqual(content, "Hello I am a fixture for testing.")

    def test_2(self):
        # another test
```
Every test class inherits from `MultiMetaSSDTestCase` class which in turn inherits from `unittest.TestCase`. Therefore it has a series of functions available to it for [assertions](https://docs.python.org/3/library/unittest.html#test-cases) and other [utility](./multi_meta_ssd/MultiMetaSSDTestCase.py).

If you need to temporarily disable a test, you can simply change the function name so that it no longer starts with `test_`.  For example, `test_example_fixture` would be skipped if you renamed it to be `do_not_test_example_fixture`.

## Entry points

All entry points of the multi_meta_ssd package are managed through a single script `multi_meta_ssd` which resides in `multi_meta_ssd/commands/main.py : main_cmd()`. Each entry point has to have a file named `create_<entry_point>_parser` (e.g., [create_action1_parser.py](multi_meta_ssd/commands/create_action1_parser.py) and [create_action2_parser.py](multi_meta_ssd/commands/create_action2_parser.py)). These then should be called in [main.py](multi_meta_ssd/commands/main.py).

You can try the current entry points
```
# Entry point 1, sub entry point 1, sub entry points are useful for organization
multi_meta_ssd action1 stuff

# Entry point 2, sub entry point 1, sub entry points are useful for organization
multi_meta_ssd action2 subaction1

# Entry point 2, sub entry point 2, sub entry points are useful for organization
multi_meta_ssd action2 subaction2 --something something
```

## Details of scripts and config files

- `tools/setup.ps1`: Run this in powershell in windows with `& tools/setup.ps1` from the project root directory before starting development. If you open a new powershell window or tab, run this script again.
- `tools/setup.cmd`: Run this in cmd.exe in windows with `call tools/setup.cmd` from the project root directory before starting development. If you open a new cmd.exe window, run this script again.
- `tools/setup.sh`: Run this in bash in mac or linux with `source tools/setup.sh` from the project root directory before starting development. If you open a new bash window,run this script again.
- `setup.py`: This is internally used by the `setup.ps1|cmd|sh` scripts. This will install your code in dev mode, so that python always sees your package when your conda env is active. Also it will install your package entry points. So for example, from the commands like `multi_meta_ssd_ep1` is always recognized and runs `multi_meta_ssd/commands/ep1.py`.
- `tools/lint.sh`: Run the python linter. It will tell you potential errors in the code. Linter config are stored in the `.pylintrc` file.
- `tool/coverage.sh`: Run your tests and also report how much of your code is covered by the tests. Coverage config are stored in the `.coveragerc` file.
- `tools/artifact.py`, `tools/tiny.yaml`, `tools/artifctory.manifest`: artifactory integration.
- `tools/conda.yaml`: requirements of the project which will be installed by conda.


# Credits

This scaffold is a fork of the [python-scaffold](https://git.corp.adobe.com/3di/python-scaffold) repo by the titans at 3DI. We simply removed the Artifactory and the Jenkins sections. Additional credits from them: 

This scaffold is greatly inspired by the [metabuild](https://git.corp.adobe.com/meta-build/meta-build) project. Artifactory integration is provided by [tiny](https://git.corp.adobe.com/lagrange/tiny).  The [3DI Tech Transfer team](https://wiki.corp.adobe.com/display/3DI/3DI+tech+transfer+team) maintains this template repo. 
