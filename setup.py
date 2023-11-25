from setuptools import setup, find_packages

# Read the README.md for long description
with open("README.md", "r") as fh:
    long_description = fh.read()

# Read the requirements from requirements_prod.txt
with open("requirements_prod.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name='zoish',
    version='5.0.2',
    author='drhosseinjavedani',
    author_email='h.javedani@gmail.com',
    description=("Zoish is a Python package that streamlines machine learning by leveraging SHAP values for feature selection and interpretability, making model development more efficient and user-friendly."),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/TorkamaniLab/zoish',
    license='BSD-3-Clause license',
    packages=find_packages(exclude=["examples*"]),
    include_package_data=True,  # This will read MANIFEST.in
    keywords=["Auto ML",
    "Feature Selection",
    "Pipeline",
    "Machine learning",
    "shap"],
    install_requires=requirements  # Use the parsed requirements here
)