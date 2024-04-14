from setuptools import setup, find_packages

# Read the contents of your requirements file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='sgarbi',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,
    author='Erick Sgarbi',
    author_email='erick.sgarbi@gmail.com',
    description='A custom text-to-text generation to generate questions and answers from a given text.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
