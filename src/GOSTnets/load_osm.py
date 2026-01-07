####################################################################################################
# Load OSM data into network graph
# Benjamin Stewart and Charles Fox
# Purpose: take an input dataset as a OSM file and return a network object
####################################################################################################

import time

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

# from geopy.distance import geodesic
from geopy import distance
from boltons.iterutils import pairwise
from shapely.wkt import loads

from . import conversion_utils as cu


class OSM_to_network(object):
    """
    Object to load OSM PBF to networkX objects.

    Example
    -------
    >>> G_loader = losm.OSM_to_network(bufferedOSM_pbf) \
    >>> G_loader.generateRoadsGDF() \
    >>> G = G.initialReadIn() \
    >>> # snap origins and destinations \
    >>> o_snapped = gn.pandana_snap(G, origins) \
    >>> d_snapped = gn.pandana_snap(G, destinations) \

    """

    def __init__(self, osmFile, includeFerries=False):
        """
        Generate a networkX object from a osm file

        Parameters
        ----------
        osmFile : string
            The path to the OSM file

        includeFerries : boolean
            Include ferries in the network

        Returns
        -------
        None

        """
        self.osmFile = osmFile
        self.roads_raw = (
            self.fetch_roads_and_ferries(osmFile)
            if includeFerries
            else self.fetch_roads(osmFile)
        )

    def generateRoadsGDF(self, in_df=None, outFile="", verbose=False):
        """
        post-process roads GeoDataFrame adding additional attributes

        Parameters
        ----------
        in_df : GeoDataFrame
            Optional input GeoDataFrame
        outFile : string
            optional parameter to output a csv with the processed roads

        Returns
        -------
        float
            Length of line in kilometers

        """
        if not isinstance(in_df, gpd.geodataframe.GeoDataFrame):
            in_df = self.roads_raw

        # get all intersections
        roads = cu.get_all_intersections(in_df, unique_id="osm_id")

        # add new key column that has a unique id
        roads["key"] = ["edge_" + str(x + 1) for x in range(len(roads))]
        np.arange(1, len(roads) + 1, 1)

        def get_nodes(x):
            return list(x.geometry.coords)[0], list(x.geometry.coords)[-1]

        # generate all of the nodes per edge and to and from node columns
        nodes = gpd.GeoDataFrame(
            roads.apply(lambda x: get_nodes(x), axis=1).apply(pd.Series)
        )
        nodes.columns = ["u", "v"]

        # compute the length per edge
        roads["length"] = roads.geometry.apply(lambda x: self.line_length(x))
        roads.rename(columns={"geometry": "Wkt"}, inplace=True)

        roads = pd.concat([roads, nodes], axis=1)

        if outFile != "":
            roads.to_csv(outFile)

        self.roadsGPD = roads

    def filterRoads(
        self,
        acceptedRoads=[
            "primary",
            "primary_link",
            "secondary",
            "secondary_link",
            "motorway",
            "motorway_link",
            "trunk",
            "trunk_link",
        ],
    ):
        """
        Extract certain times of roads from the OSM before the netowrkX conversion

        Parameters
        ----------
        acceptedRoads : list of strings, optional
            list of accepted road types

        Returns
        -------
        None
            the raw roads are filtered based on the list of accepted roads

        """
        self.roads_raw = self.roads_raw.loc[
            self.roads_raw.infra_type.isin(acceptedRoads)
        ]

    def fetch_roads(self, data_path):
        """
        Extracts roads from an OSM PBF

        Parameters
        ----------
        data_path : string
            The directory of the shapefiles consisting of edges and nodes

        Returns
        -------
        GeoDataFrame
            a road GeoDataFrame

        """
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"OSM file not found: {path}")
        if path.suffix.lower() == ".pbf":
            driver = ogr.GetDriverByName("OSM")
            data = driver.Open(str(path))
            if data is None:
                raise ValueError(f"OGR could not open OSM file: {path}")
            sql_lyr = data.ExecuteSQL(
                "SELECT osm_id, highway, other_tags FROM lines WHERE highway IS NOT NULL"
            )
            roads = []

            for feature in sql_lyr:
                if feature.GetField("highway") is not None:
                    osm_id = feature.GetField("osm_id")
                    shapely_geo = loads(feature.geometry().ExportToWkt())
                    if shapely_geo is None:
                        continue
                    highway = feature.GetField("highway")

                    if feature.GetField("other_tags"):
                        other_tags = feature.GetField("other_tags")
                        # print("print other tags")
                        # print(other_tags)

                        # there may be rare cases where there can be a comma within the value of an other tag, if so we just have to skip
                        try:
                            other_tags_dict = dict(
                                (x.strip('"'), y.strip('"'))
                                for x, y in (
                                    element.split("=>")
                                    for element in other_tags.split(",")
                                )
                            )

                            if other_tags_dict.get("oneway") == "yes":
                                one_way = True
                            else:
                                one_way = False
                        except Exception:
                            print(
                                f"skipping over reading other tags of osm_id: {osm_id}"
                            )
                            one_way = False

                    else:
                        one_way = False
                    roads.append([osm_id, highway, one_way, shapely_geo])

            if len(roads) > 0:
                road_gdf = gpd.GeoDataFrame(
                    roads,
                    columns=["osm_id", "infra_type", "one_way", "geometry"],
                    crs="epsg:4326",
                )
                return road_gdf

        elif path.suffix.lower() == ".shp":
            road_gdf = gpd.read_file(path)
            return road_gdf

        else:
            print("No roads found")

    def fetch_roads_and_ferries(self, data_path):
        """
        Extracts roads and ferries from an OSM PBF

        Parameters
        ----------
        data_path : string
            The directory of the shapefiles consisting of edges and nodes

        Returns
        -------
        GeoDataFrame
            a road GeoDataFrame

        """
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"OSM file not found: {path}")
        if path.suffix.lower() == ".pbf":
            driver = ogr.GetDriverByName("OSM")
            data = driver.Open(str(path))
            if data is None:
                raise ValueError(f"OGR could not open OSM file: {path}")
            sql_lyr = data.ExecuteSQL("SELECT * FROM lines")

            roads = []

            for feature in sql_lyr:
                if feature.GetField("man_made"):
                    if "pier" in feature.GetField("man_made"):
                        osm_id = feature.GetField("osm_id")
                        shapely_geo = loads(feature.geometry().ExportToWkt())
                        if shapely_geo is None:
                            continue
                        highway = "pier"
                        roads.append([osm_id, highway, shapely_geo])
                elif feature.GetField("other_tags"):
                    if "ferry" in feature.GetField("other_tags"):
                        osm_id = feature.GetField("osm_id")
                        shapely_geo = loads(feature.geometry().ExportToWkt())
                        if shapely_geo is None:
                            continue
                        highway = "ferry"
                        roads.append([osm_id, highway, shapely_geo])
                    elif feature.GetField("highway") is not None:
                        osm_id = feature.GetField("osm_id")
                        shapely_geo = loads(feature.geometry().ExportToWkt())
                        if shapely_geo is None:
                            continue
                        highway = feature.GetField("highway")
                        roads.append([osm_id, highway, shapely_geo])
                elif feature.GetField("highway") is not None:
                    osm_id = feature.GetField("osm_id")
                    shapely_geo = loads(feature.geometry().ExportToWkt())
                    if shapely_geo is None:
                        continue
                    highway = feature.GetField("highway")
                    roads.append([osm_id, highway, shapely_geo])

            data = driver.Open(str(path))
            sql_lyr_ferries = data.ExecuteSQL(
                "SELECT * FROM multipolygons WHERE multipolygons.amenity = 'ferry_terminal'"
            )

            for feature in sql_lyr_ferries:
                osm_id = feature.GetField("osm_id")
                shapely_geo = shapely.ops.linemerge(
                    loads(feature.geometry().ExportToWkt()).boundary
                )
                if shapely_geo is None:
                    continue
                highway = "pier"
                roads.append([osm_id, highway, shapely_geo])

            if len(roads) > 0:
                road_gdf = gpd.GeoDataFrame(
                    roads,
                    columns=["osm_id", "infra_type", "geometry"],
                    crs="epsg:4326",
                )
                return road_gdf

        elif path.suffix.lower() == ".shp":
            road_gdf = gpd.read_file(path)
            return road_gdf

        else:
            print("No roads found")

    def line_length(self, line, ellipsoid="WGS-84"):
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
            return sum(self.line_length(segment) for segment in line)

        return sum(
            distance.geodesic(
                tuple(reversed(a)), tuple(reversed(b)), ellipsoid=ellipsoid
            ).km
            for a, b in pairwise(line.coords)
        )

    def get_all_intersections(
        self, shape_input, idx_osm=None, unique_id="osm_id", verboseness=False
    ):
        """
        Processes GeoDataFrame and splits edges as intersections

        Parameters
        ----------
        shape_input : GeoDataFrame
            Input GeoDataFrame
        idx_osm : spatial index
            The geometry index name
        unique_id : string
            The unique id field name

        Returns
        -------
        GeoDataFrame
            returns processed GeoDataFrame

        """


    def initialReadIn(self, fpath=None, wktField="Wkt"):
        """
        Convert the OSM object to a networkX object

        Parameters
        ----------
        fpath : string
            path to CSV file with roads to read in
        wktField : string
            wktField name

        Returns
        -------
        nx.MultiDiGraph
            a networkX MultiDiGraph object

        """
        if isinstance(fpath, str):
            edges_1 = pd.read_csv(fpath)
            edges_1 = edges_1[wktField].apply(lambda x: loads(x))
        elif isinstance(fpath, gpd.GeoDataFrame):
            edges_1 = fpath
        else:
            try:
                edges_1 = self.roadsGPD
            except Exception:
                self.generateRoadsGDF()
                edges_1 = self.roadsGPD

        edges = edges_1.copy()
        node_bunch = list(set(list(edges["u"]) + list(edges["v"])))

        def convert(x):
            u = x.u
            v = x.v
            data = {
                "Wkt": x.Wkt,
                "id": x.id,
                "infra_type": x.infra_type,
                "one_way": x.one_way,
                "osm_id": x.osm_id,
                "key": x.key,
                "length": x.length,
            }
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
        self.network = G

        return G
