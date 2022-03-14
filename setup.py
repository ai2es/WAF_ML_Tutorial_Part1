from setuptools import setup, find_packages
import os

# For guidance on setuptools best practices visit
# https://packaging.python.org/guides/distributing-packages-using-setuptools/
project_name = os.getcwd().split("/")[-1]
version = "0.1.0"
package_description = "<Provide short description of package>"
url = "https://github.com/ai2es/" + project_name
# Classifiers listed at https://pypi.org/classifiers/
classifiers = ["Programming Language :: Python :: 3"]
setup(name="ai2estemplate", # Change
      version=version,
      description=package_description,
      url=url,
      author="AI2ES",
      license="CC0 1.0",
      classifiers=classifiers,
      packages=find_packages(include=["src"]))

