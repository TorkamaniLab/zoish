from setuptools import setup, find_packages
from pathlib import Path
import sys

root = Path(__file__).parent
long_description = (root / "README.md").read_text(encoding="utf-8")
req_file = root / "requirements_prod.txt"
requirements = req_file.read_text(encoding="utf-8").splitlines() if req_file.exists() else []

# Optional: a clearer error if someone tries Python 3.13+
if sys.version_info >= (3, 13):
    raise SystemExit(
        "zoish requires Python < 3.13 because 'fasttreeshap/numba/llvmlite' "
        "do not yet provide wheels for 3.13. "
        "Please use Python 3.8–3.12."
    )

setup(
    name='zoish',
    version='5.0.8',
    author='drhosseinjavedani',
    author_email='h.javedani@gmail.com',
    description=("Zoish streamlines ML with SHAP-based feature selection & interpretability."),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/TorkamaniLab/zoish',
    license='BSD-3-Clause license',
    packages=find_packages(exclude=["examples*"]),
    include_package_data=True,
    keywords=["Auto ML", "Feature Selection", "Pipeline", "Machine learning", "shap"],
    install_requires=requirements,
    python_requires='>=3.8,<3.13',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
