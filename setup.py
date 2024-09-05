from setuptools import setup, find_packages

setup(
    name='pyDRReg',
    version='0.1',
    description='A package for Doubly Robust estimation and other estimators.',
    author='Daniel de Abreu Pereira Uhr',
    author_email='daniel.uhr@gmail.com',
    url='https://github.com/Daniel-Uhr/pyDRReg',
    packages=find_packages(),  # Automaticamente encontra a subpasta pyDRReg como um pacote
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'statsmodels'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
