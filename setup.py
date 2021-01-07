from setuptools import setup, find_packages

setup(
    name="GOSTnets",
    version="1.0.0",
    packages=find_packages(),

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=[#'docutils>=0.3',
                      'geopandas>=0.4.0',
                      'networkx',
                      'numpy',
                      'osmnx',
                      'pandas>=0.23.5',
                      'pyproj',
                      'scipy',
                      'shapely'
                      ],

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        # And include any *.msg files found in the 'hello' package, too:
    },

    # metadata to display on PyPI
    author="Benjamin P. Stewart",
    author_email="ben.gis.stewart@gmail.com",
    description="Networkx wrapper to simplify network analysis using geospatial data",
    license="PSF",
    keywords="networkx networks OSM",
    url="https://github.com/worldbank/GOSTnets",   # project home page, if any
    project_urls={
        "Bug Tracker": "https://github.com/worldbank/GOSTnets/issues",
        "Documentation": "https://github.com/worldbank/GOSTnets",
        "Source Code": "https://github.com/worldbank/GOSTnets",
    }

    # could also include long_description, download_url, classifiers, etc.
)