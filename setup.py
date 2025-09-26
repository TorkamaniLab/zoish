from setuptools import setup, find_packages
from pathlib import Path

root = Path(__file__).parent

long_description = (root / "README.md").read_text(encoding="utf-8")

req_file = root / "requirements_prod.txt"
if req_file.exists():
    requirements = req_file.read_text(encoding="utf-8").splitlines()
else:
    requirements = []  # fallback so build doesn't crash

setup(
    name='zoish',
    version='5.0.6',
    author='drhosseinjavedani',
    author_email='h.javedani@gmail.com',
    description=("Zoish is a Python package that streamlines machine learning by leveraging SHAP values for feature selection and interpretability, making model development more efficient and user-friendly."),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/TorkamaniLab/zoish',
    license='BSD-3-Clause license',
    packages=find_packages(exclude=["examples*"]),
    include_package_data=True,
    keywords=["Auto ML","Feature Selection","Pipeline","Machine learning","shap"],
    install_requires=requirements,
)
