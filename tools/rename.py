import argparse
import logging
import shutil
from pathlib import Path

logger = logging.getLogger("rename")
logger.setLevel("INFO")
logger.addHandler(logging.StreamHandler())
logger.handlers[0].setFormatter(logging.Formatter("%(message)s"))


def main():

    parser = argparse.ArgumentParser("Script to rename everything to your new name")
    parser.add_argument("--snake-case", type=str, help="snake case version of your new name (e.g., awesome_lib)", required=True)
    parser.add_argument("--camel-case", type=str, help="camel case version of your new name (e.g., AwesomeLib)", required=True)
    parser.add_argument("--caps-case", type=str, help="all caps version of your new name (e.g., AWESOME_LIB)", required=True)
    args = parser.parse_args()
    new_name = args.snake_case
    NewName = args.camel_case
    NEWNAME = args.caps_case

    root = Path(__file__).absolute().parents[1]
    FILES = [
        *list(root.glob("*.*")),
        *list(root.glob(".vscode/**/*.*")),
        *list(root.glob("tests/**/*.*")),
        *list(root.glob("thrall/**/*.*")),
        *list(root.glob("tools/*.*")),
    ]

    for f in FILES:
        if f.stem.startswith("."):
            continue
        if f.is_dir():
            continue
        if f.relative_to(root).as_posix() == "tools/rename.py":
            continue
        logger.info("processing %s", f)
        with f.open("r") as stream:
            try:
                content = stream.read()
            except UnicodeDecodeError:
                logger.info("file is junk, continue")
                continue
        content = content.replace("thrall", new_name)
        content = content.replace("Thrall", NewName)
        content = content.replace("THRALL", NEWNAME)
        with f.open("w") as stream:
            stream.write(content)

    logger.info("renaming folder")
    if (root / "thrall" / "thrall_test_case.py").exists():
        shutil.move(root / "thrall" / "thrall_test_case.py", root / "thrall" / f"{new_name}_test_case.py")
    if (root / "thrall").exists():
        shutil.move(root / "thrall", root / new_name)


if __name__ == "__main__":
    main()
