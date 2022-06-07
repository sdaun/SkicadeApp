from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='skin-cancer-detection',
      version="1.0",
      description="Using ML to predict the classes of skin lesions and detect skin cancer",
      packages=find_packages(),
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      install_requires=requirements)
