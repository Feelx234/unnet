from setuptools import setup

setup(
    name='unnet',
    version='0.0.1',
    packages=['unnet'],
    url='',
    license='',
    author='Felix Stamm',
    author_email='felix.stamm@cssh.rwth-aachen.de',
    description='un-net is a library that can be used to study uncertainty in networks',
    install_requires=[
        'pandas>=0.24.0',
        'numpy',
        'matplotlib',
#        'graph-tool',
        'numba',
        'tqdm',
    ],
    tests_require=['pytest'],
    python_requires='>=3.6',
    classifiers=['Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'Programming Language :: Python',
                 'Topic :: Software Development',
                 'Topic :: Scientific/Engineering',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: POSIX',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 ('Programming Language :: Python :: '
                  'Implementation :: CPython')
                 ],
)
