from distutils.core import setup

setup(
    name='colibri',
    version='0.1.0',
    author='Gabriele Parimbelli',
    author_email='g.parimbelli90@gmail.com',
    packages=['colibri'],
    scripts=[],
    url='',
    license='doc/source/license/license.rst',
    description='Python libraries for cosmology.',
    long_description=open('README.rst').read(),
    install_requires=[
        "numpy >= 1.16",
        "scipy >= 1.0",
    ],
)
