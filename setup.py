from setuptools import setup, find_packages

version = {}
with open('delfta/__version__.py') as fp:
    exec(fp.read(), version)

    
setup(name='pyGPGO',
    version=version['__version__'],
    description='delfta',
    classifiers=[
    'Development Status :: 2 - Pre-Alpha',
    'Topic :: Scientific/Engineering :: Mathematics'
    ],
    keywords = [],
    url='',
    author='',
    author_email='',
    license='',
    packages=find_packages(),
    zip_safe=False)