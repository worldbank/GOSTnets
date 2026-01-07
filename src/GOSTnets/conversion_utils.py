import time
import rasterio
import shapely.ops

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
# import matplotlib.pyplot as plt

from osgeo import ogr
from rtree import index
from shapely.geometry import LineString, MultiPoint
from pathlib import Path
from scipy import interpolate
from rasterio import features

# from geopy.distance import geodesic
from geopy import distance
from boltons.iterutils import pairwise
from shapely.wkt import loads

def get_all_intersections(shape_input, unique_id, infra_field="infra_type"):
    """Get all intersections in a linestring geodataframe and cut lines at intersections
    Parameters
    ----------
    shape_input : GeoDataFrame
        input linestring geodataframe
    unique_id : string
        unique id field in geodataframe
    infra_field : string
        infrastructure type field in geodataframe

    Returns
    -------
    GeoDataFrame
        GeoDataFrame with all lines cut at intersections
    """
    # Initialize Rtree
    idx_inters = index.Index()
    idx_osm = shape_input["geometry"].sindex

    # Find all the intersecting lines to prepare for cutting
    count = 0
    tLength = shape_input.shape[0]
    start = time.time()
    inters_done = {}
    new_lines = []
    # allCounts = []

    for idx, row in shape_input.iterrows():
        # print(row)
        key1 = row[f"{unique_id}"]
        line = row.geometry
        infra_type = row[infra_field]
        one_way = row.get("one_way")
        if (count % 1000 == 0):
            print("Processing %s of %s" % (count, tLength))
            print("seconds elapsed: " + str(time.time() - start))
        count += 1
        intersections = shape_input.iloc[list(idx_osm.intersection(line.bounds))]
        intersections = dict(
            zip(list(intersections[f"{unique_id}"]), list(intersections.geometry))
        )
        if key1 in intersections:
            intersections.pop(key1)
        # Find intersecting lines
        for key2, line2 in intersections.items():
            # Check that this intersection has not been recorded already
            if (key1, key2) in inters_done or (key2, key1) in inters_done:
                continue
            # Record that this intersection was saved
            inters_done[(key1, key2)] = True
            # Get intersection
            if line.intersects(line2):
                # Get intersection
                inter = line.intersection(line2)
                # Save intersecting point
                # updating to be compatible with Shapely ver 2
                # if "Point" == inter.type:
                if "Point" == inter.geom_type:
                    idx_inters.insert(0, inter.bounds, inter)
                elif "MultiPoint" == inter.geom_type:
                    # updating to be compatible with Shapely ver 2
                    # for pt in inter:
                    for pt in inter.geoms:
                        idx_inters.insert(0, pt.bounds, pt)

        # cut lines where necessary and save all new linestrings to a list
        hits = [
            n.object for n in idx_inters.intersection(line.bounds, objects=True)
        ]

        if len(hits) != 0:
            try:
                out = shapely.ops.split(line, MultiPoint(hits))
                new_lines.append(
                    [
                        {
                            "geometry": LineString(x),
                            "osm_id": key1,
                            "infra_type": infra_type,
                            "one_way": one_way,
                        }
                        for x in out.geoms
                    ]
                )
            except Exception:
                pass
        else:
            new_lines.append(
                [
                    {
                        "geometry": line,
                        "osm_id": key1,
                        "infra_type": infra_type,
                        "one_way": one_way,
                    }
                ]
            )

    # Create one big list and treat all the cut lines as unique lines
    flat_list = []
    all_data = {}

    # item for sublist in new_lines for item in sublist
    i = 1
    for sublist in new_lines:
        if sublist is not None:
            for item in sublist:
                item["id"] = i
                flat_list.append(item)
                i += 1
                all_data[i] = item

    # Transform into geodataframe and add coordinate system
    full_gpd = gpd.GeoDataFrame(flat_list, geometry="geometry", crs=shape_input.crs)
    return full_gpd

def line_length(line, ellipsoid="WGS-84"):
    """
    Returns length of a line in kilometers, given in geographic coordinates. Adapted from https://gis.stackexchange.com/questions/4022/looking-for-a-pythonic-way-to-calculate-the-length-of-a-wkt-linestring#answer-115285

    Parameters
    ----------
    line : LineString
        a shapely LineString object with WGS-84 coordinates
    ellipsoid : str
        string name of an ellipsoid that `geopy` understands (see http://geopy.readthedocs.io/en/latest/#module-geopy.distance)

    Returns
    -------
    float
        Length of line in kilometers

    """
    if line.geom_type == "MultiLineString":
        return sum(line_length(segment) for segment in line)

    return sum(
        distance.geodesic(
            tuple(reversed(a)), tuple(reversed(b)), ellipsoid=ellipsoid
        ).km
        for a, b in pairwise(line.coords)
    )

def generateRoadsGDF(in_df, unq_id="osm_id", infra_field="infra_type"):
    """
    post-process roads GeoDataFrame adding additional attributes

    Parameters
    ----------
    in_df : GeoDataFrame
        input GeoDataFrame
    
    Returns
    -------
    float
        Length of line in kilometers

    """
    # get all intersections
    roads = get_all_intersections(in_df, unique_id=unq_id, infra_field=infra_field)

    # add new key column that has a unique id
    roads["key"] = ["edge_" + str(x + 1) for x in range(len(roads))]
    np.arange(1, len(roads) + 1, 1)

    def get_nodes(x):
        return list(x.geometry.coords)[0], list(x.geometry.coords)[-1]

    # generate all of the nodes per edge and to and from node columns
    nodes = gpd.GeoDataFrame(roads.apply(lambda x: get_nodes(x), axis=1).apply(pd.Series))
    nodes.columns = ["u", "v"]

    # compute the length per edge
    roads["length"] = roads.geometry.apply(lambda x: line_length(x))
    roads.rename(columns={"geometry": "Wkt"}, inplace=True)

    roads = pd.concat([roads, nodes], axis=1)

    return(roads)

def convert_roads_to_nx(edges_1, unq_id="osm_id", infra_field="infra_type"):
    """Convert roads GeoDataFrame to NetworkX MultiDiGraph
    Parameters
    ----------
    edges_1 : GeoDataFrame
        input roads GeoDataFrame
    cols_to_keep : list
        list of columns to keep in edge attributes
    unq_id : string
        unique id field in geodataframe
    infra_field : string
        infrastructure type field in geodataframe

    Returns
    -------
    NetworkX MultiDiGraph
        For use in GOSTnets graph calculations
    """
    edges = generateRoadsGDF(edges_1, unq_id=unq_id, infra_field=infra_field)
    node_bunch = list(set(list(edges["u"]) + list(edges["v"])))

    def convert(x):
        u = x.u
        v = x.v
        data = {}
        for col in ['infra_type', 'one_way', 'osm_id', 'key', 'length', 'Wkt']:
            data[col] = x[col]        
        return (u, v, data)

    edge_bunch = edges.apply(lambda x: convert(x), axis=1).tolist()

    G = nx.MultiDiGraph()
    G.add_nodes_from(node_bunch)
    G.add_edges_from(edge_bunch)

    for u, data in G.nodes(data=True):
        if isinstance(u, str):
            q = tuple(float(x) for x in u[1:-1].split(","))
        if isinstance(u, tuple):
            q = u
        data["x"] = q[0]
        data["y"] = q[1]
    G = nx.convert_node_labels_to_integers(G)
    return G

def rasterize_od_results(inD, outFile, field, template=None):
    """Convert gridded point data frame to raster of commensurate size and resolution

    Parameters
    ----------
    inD : geopandas data frame
        OD matrix as point data frame
    outFile: string
        path to save output raster
    field : string
        field to rasterize
    template : string, optional
        path to raster template file, by default None

    Returns
    -------
    None

    """
    if template:
        raster_template = rasterio.open(template)
        # get info from template file
        xRes = raster_template.res[1]
        yRes = raster_template.res[0]
        trans = raster_template.transform
        x_pixels = raster_template.shape[1]
        y_pixels = raster_template.shape[0]

        new_dataset = rasterio.open(
            outFile,
            "w",
            driver="GTiff",
            height=y_pixels,
            width=x_pixels,
            count=1,
            dtype=str(inD[field].dtype),
            crs=raster_template.crs,
            transform=trans,
        )

        shapes = ((row.geometry, row[field]) for idx, row in inD.iterrows())
        burned = features.rasterize(
            shapes=shapes, fill=0, out_shape=raster_template.shape, transform=trans
        )
        burned = burned.astype(str(inD[field].dtype))
        new_dataset.write_band(1, burned)

        new_dataset.close()

    else:
        # create grid from input shapefile
        # get xs, ys, and values from origin points
        xs = np.array(inD.geometry.apply(lambda p: p.x))
        ys = np.array(inD.geometry.apply(lambda p: p.y))
        vals = np.array(inD[field])
        # creates a full grid for the entire bounding box (all pairs of xs and ys)
        unique_xs = np.unique(xs)
        unique_ys = np.unique(ys)
        xx, yy = np.meshgrid(unique_xs, unique_ys)
        # this creates a new set of values to fill the grid
        grid_array = interpolate.griddata((xs, ys), vals, (xx, yy))
        x_pixels = grid_array.shape[1]
        y_pixels = grid_array.shape[0]
        # get the right transformation for raster file
        xRes = (xx.max() - xx.min()) / len(unique_xs)
        yRes = (yy.max() - yy.min()) / len(unique_ys)
        # get the right transformation for raster file
        trans = rasterio.transform.from_bounds(
            xx.min() - (xRes / 2),
            yy.min() - (yRes / 2),
            xx.max() - (xRes / 2),
            yy.max() - (yRes / 2),
            x_pixels - 1,
            y_pixels - 1,
        )
        new_dataset = rasterio.open(
            outFile,
            "w",
            driver="GTiff",
            height=y_pixels,
            width=x_pixels,
            count=1,
            dtype=str(grid_array.dtype),
            crs=inD.crs,
            transform=trans,
        )

        shapes = ((row.geometry, row[field]) for idx, row in inD.iterrows())
        burned = features.rasterize(
            shapes=shapes,
            fill=0,
            out_shape=grid_array.shape,
            transform=new_dataset.transform,
        )
        burned = burned.astype(grid_array.dtype)
        new_dataset.write_band(1, burned)

        new_dataset.close()
