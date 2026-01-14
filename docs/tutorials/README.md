# GOSTnets Tutorials

These notebooks are designed to introduce the methodology of GOSTnets to novice Python users; in order to properly implement these, users should have a basic knowledge of

- Python
- Jupyter notebooks
- Anaconda (for setup)
- GIS/Geography

# Installation
Make sure you install GOSTnets using the tutorials extension to support Jupyter Notebooks

The first line includes conda install of several libraries; this is to support Windows users for whom gdal and geopandas do not install cleanly through pip.
```
conda create --name gostnets geopandas gdal osmnx -c conda-forge
conda activate gostnets
pip install GOSTnets[tutorials]
```

# Outline
There are several notebooks in the GOSTnets tutorial, choose the one that is right for you:

| Name | Description |
| --- | --- |
| Step_1-Extract_road_network.ipynb | Explores how to import road networks from OSM and Overture |
| Step_2-Clean_Network.ipynb | Walks through the network cleaning process |
| Step_3-Using_your_Graph.ipynb | Explores basics of running graph analysis |
| EXAMPLE_Finding_links_between_pairs.ipynb | Walks through a simple example of mesauring travel time |
| EXAMPLE_Fixing_your_road_network.ipynb | Explores how to fix topological errors in your road network |
| EXAMPLE_Gravity_Calculations.ipynb | Explores the concept of gravity, and how it is calculated in GOSTnets |
