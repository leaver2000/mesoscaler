from setuptools import setup, Extension
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
PROJECT_NAME = "mesoscaler"
PROJECT_DIR = ROOT / "src" / PROJECT_NAME


def find_extension(module_name: str, suffixes: list[str], **kwargs):
    module = PROJECT_DIR / module_name
    module_name = f"{PROJECT_NAME}.{module_name}"
    sources = [str(file.relative_to(ROOT)) for file in module.rglob("*") if file.suffix in suffixes]
    return Extension(module_name, sources, **kwargs)


extension_kwargs = {
    "include_dirs": [np.get_include()],
    "extra_compile_args": ["-O0", "-g", "-fprofile-arcs", "-ftest-coverage", "--coverage"],
    "extra_link_args": ["-fprofile-arcs", "-ftest-coverage"],
}


extension_modules = [find_extension("_C", [".c", ".cpp"], **extension_kwargs)]

setup(ext_modules=extension_modules)
