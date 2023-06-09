####################################################################################################
# Load OSM data into network graph
# Benjamin Stewart and Charles Fox
# Purpose: take an input dataset as a OSM file and return a network object
####################################################################################################

import os, sys, time

import shapely.ops

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
# import matplotlib.pyplot as plt

from osgeo import ogr
from rtree import index
from shapely import speedups
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point
#from geopy.distance import geodesic
from geopy import distance
from boltons.iterutils import pairwise
from shapely.wkt import loads,dumps

class OSM_to_network(object):
    """
    Object to load OSM PBF to networkX objects.

    Object to load OSM PBF to networkX objects. \
    EXAMPLE: \
    G_loader = losm.OSM_to_network(bufferedOSM_pbf) \
    G_loader.generateRoadsGDF() \
    G = G.initialReadIn() \

    snap origins and destinations \
    o_snapped = gn.pandana_snap(G, origins) \
    d_snapped = gn.pandana_snap(G, destinations) \
    """

    def __init__(self, osmFile, includeFerries=False):
        """
        Generate a networkX object from a osm file
        """
        self.osmFile = osmFile
        self.roads_raw = self.fetch_roads_and_ferries(osmFile) if includeFerries else self.fetch_roads(osmFile)

    def generateRoadsGDF(self, in_df = None, outFile='', verbose = False):
        """
        post-process roads GeoDataFrame adding additional attributes

        :param in_df: Optional input GeoDataFrame
        :param outFile: optional parameter to output a csv with the processed roads
        :returns: Length of line in kilometers
        """
        if type(in_df) != gpd.geodataframe.GeoDataFrame:
            in_df = self.roads_raw

        # get all intersections
        roads = self.get_all_intersections(in_df, unique_id = 'osm_id', verboseness = verbose)

        # add new key column that has a unique id
        roads['key'] = ['edge_'+str(x+1) for x in range(len(roads))]
        np.arange(1,len(roads)+1,1)

        def get_nodes(x):
            return list(x.geometry.coords)[0],list(x.geometry.coords)[-1]

        # generate all of the nodes per edge and to and from node columns
        nodes = gpd.GeoDataFrame(roads.apply(lambda x: get_nodes(x),axis=1).apply(pd.Series))
        nodes.columns = ['u','v']

        # compute the length per edge
        roads['length'] = roads.geometry.apply(lambda x : self.line_length(x))
        roads.rename(columns={'geometry':'Wkt'}, inplace=True)

        roads = pd.concat([roads,nodes],axis=1)

        if outFile != '':
            roads.to_csv(outFile)

        self.roadsGPD = roads

    def filterRoads(self, acceptedRoads = ['primary','primary_link','secondary','secondary_link','motorway','motorway_link','trunk','trunk_link']):
        """
        Extract certain times of roads from the OSM before the netowrkX conversion

        :param acceptedRoads: [ optional ] acceptedRoads [ list of strings ]
        :returns: None - the raw roads are filtered based on the list of accepted roads
        """

        self.roads_raw = self.roads_raw.loc[self.roads_raw.infra_type.isin(acceptedRoads)]

    def fetch_roads(self, data_path):
        """
        Extracts roads from an OSM PBF

        :param data_path: The directory of the shapefiles consisting of edges and nodes
        :returns: a road GeoDataFrame
        """

        if data_path.split('.')[-1] == 'pbf':
            driver = ogr.GetDriverByName("OSM")
            data = driver.Open(data_path)
            sql_lyr = data.ExecuteSQL("SELECT osm_id, highway, other_tags FROM lines WHERE highway IS NOT NULL")
            roads = []

            for feature in sql_lyr:
                if feature.GetField("highway") is not None:
                    osm_id = feature.GetField("osm_id")
                    shapely_geo = loads(feature.geometry().ExportToWkt())
                    if shapely_geo is None:
                        continue
                    highway = feature.GetField("highway")
                                        
                    if feature.GetField('other_tags'):
                        other_tags = feature.GetField('other_tags')
                        # print("print other tags")
                        # print(other_tags)

                        # there may be rare cases where there can be a comma within the value of an other tag, if so we just have to skip
                        try:
                            other_tags_dict = dict((x.strip('"'), y.strip('"'))
                                for x, y in (element.split('=>') 
                                for element in other_tags.split(',')))

                            if other_tags_dict.get('oneway') == 'yes':
                                one_way = True
                            else:
                                one_way = False
                        except:
                            print(f"skipping over reading other tags of osm_id: {osm_id}")
                            one_way = False          
                        
                            
                    roads.append([osm_id,highway,one_way,shapely_geo])

            if len(roads) > 0:
                road_gdf = gpd.GeoDataFrame(roads,columns=['osm_id','infra_type', 'one_way','geometry'],crs={'init': 'epsg:4326'})
                return road_gdf

        elif data_path.split('.')[-1] == 'shp':
            road_gdf = gpd.read_file(data_path)
            return road_gdf

        else:
            print('No roads found')

    def fetch_roads_and_ferries(self, data_path):
        """
        Extracts roads and ferries from an OSM PBF

        :param data_path: The directory of the shapefiles consisting of edges and nodes
        :returns: a road GeoDataFrame
        """

        if data_path.split('.')[-1] == 'pbf':

            driver = ogr.GetDriverByName('OSM')
            data = driver.Open(data_path)
            sql_lyr = data.ExecuteSQL("SELECT * FROM lines")

            roads=[]

            for feature in sql_lyr:
                if feature.GetField('man_made'):
                    if "pier" in feature.GetField('man_made'):
                        osm_id = feature.GetField('osm_id')
                        shapely_geo = loads(feature.geometry().ExportToWkt())
                        if shapely_geo is None:
                            continue
                        highway = 'pier'
                        roads.append([osm_id,highway,shapely_geo])
                elif feature.GetField('other_tags'):
                    if "ferry" in feature.GetField('other_tags'):
                        osm_id = feature.GetField('osm_id')
                        shapely_geo = loads(feature.geometry().ExportToWkt())
                        if shapely_geo is None:
                            continue
                        highway = 'ferry'
                        roads.append([osm_id,highway,shapely_geo])
                    elif feature.GetField('highway') is not None:
                        osm_id = feature.GetField('osm_id')
                        shapely_geo = loads(feature.geometry().ExportToWkt())
                        if shapely_geo is None:
                            continue
                        highway = feature.GetField('highway')
                        roads.append([osm_id,highway,shapely_geo])
                elif feature.GetField('highway') is not None:
                    osm_id = feature.GetField('osm_id')
                    shapely_geo = loads(feature.geometry().ExportToWkt())
                    if shapely_geo is None:
                        continue
                    highway = feature.GetField('highway')
                    roads.append([osm_id,highway,shapely_geo])

            data = driver.Open(data_path)
            sql_lyr_ferries = data.ExecuteSQL("SELECT * FROM multipolygons WHERE multipolygons.amenity = 'ferry_terminal'")

            for feature in sql_lyr_ferries:
                osm_id = feature.GetField('osm_id')
                shapely_geo = ops.linemerge(loads(feature.geometry().ExportToWkt()).boundary)
                if shapely_geo is None:
                    continue
                highway = 'pier'
                roads.append([osm_id,highway,shapely_geo])

            if len(roads) > 0:
                road_gdf = gpd.GeoDataFrame(roads,columns=['osm_id','infra_type','geometry'],crs={'init': 'epsg:4326'})
                return road_gdf

        elif data_path.split('.')[-1] == 'shp':
            road_gdf = gpd.read_file(data_path)
            return road_gdf

        else:
            print('No roads found')

    def line_length(self, line, ellipsoid='WGS-84'):
        """
        Returns length of a line in kilometers, given in geographic coordinates. Adapted from https://gis.stackexchange.com/questions/4022/looking-for-a-pythonic-way-to-calculate-the-length-of-a-wkt-linestring#answer-115285

        :param line: a shapely LineString object with WGS-84 coordinates
        :param string ellipsoid: string name of an ellipsoid that `geopy` understands (see http://geopy.readthedocs.io/en/latest/#module-geopy.distance)
        :returns: Length of line in kilometers
        """

        if line.geometryType() == 'MultiLineString':
            return sum(line_length(segment) for segment in line)

        return sum(
                    distance.geodesic(tuple(reversed(a)), tuple(reversed(b)), ellipsoid=ellipsoid).km
                    for a, b in pairwise(line.coords)
        )

    def get_all_intersections(self, shape_input, idx_osm = None, unique_id = 'osm_id', verboseness = False):
        """
        Processes GeoDataFrame and splits edges as intersections

        :param shape_input: Input GeoDataFrame
        :param idx_osm: The geometry index name
        :param unique_id: The unique id field name
        :returns: returns processed GeoDataFrame
        """

        # Initialize Rtree
        idx_inters = index.Index()
        # Load data
        # all_data = dict(zip(list(shape_input.osm_id),list(shape_input.geometry),list(shape_input.infra_type)))
        ### TODO - it shouldn't be necessary to reference the geometry column specifically
        #   ... but here we are

        if idx_osm is None:
            idx_osm = shape_input['geometry'].sindex

        # Find all the intersecting lines to prepare for cutting
        count = 0
        tLength = shape_input.shape[0]
        start = time.time()
        inters_done = {}
        new_lines = []
        allCounts = []

        for idx, row in shape_input.iterrows():
            #print(row)
            key1 = row[f'{unique_id}']
            line = row.geometry
            infra_type = row.infra_type
            one_way = row.get('one_way')
            if count % 1000 == 0 and verboseness == True:
                print("Processing %s of %s" % (count, tLength))
                print('seconds elapsed: ' + str(time.time() - start))
            count += 1
            intersections = shape_input.iloc[list(idx_osm.intersection(line.bounds))]
            intersections = dict(zip(list(intersections[f'{unique_id}']),list(intersections.geometry)))
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
                    #if "Point" == inter.type:
                    if "Point" == inter.type:
                        idx_inters.insert(0, inter.bounds, inter)
                    elif "MultiPoint" == inter.type:
                        # updating to be compatible with Shapely ver 2
                        #for pt in inter:
                        for pt in inter.geoms:
                            idx_inters.insert(0, pt.bounds, pt)

            # cut lines where necessary and save all new linestrings to a list
            hits = [n.object for n in idx_inters.intersection(line.bounds, objects=True)]

            if len(hits) != 0:
                try:
                    out = shapely.ops.split(line, MultiPoint(hits))
                    new_lines.append([{'geometry': LineString(x), 'osm_id':key1,'infra_type':infra_type, 'one_way':one_way} for x in out.geoms])
                except:
                    pass
            else:
                new_lines.append([{'geometry': line, 'osm_id':key1,
                        'infra_type':infra_type,'one_way':one_way}])

        # Create one big list and treat all the cutted lines as unique lines
        flat_list = []
        all_data = {}

        # item for sublist in new_lines for item in sublist
        i = 1
        for sublist in new_lines:
            if sublist is not None:
                for item in sublist:
                    item['id'] = i
                    flat_list.append(item)
                    i += 1
                    all_data[i] = item

        # Transform into geodataframe and add coordinate system
        full_gpd = gpd.GeoDataFrame(flat_list, geometry ='geometry')
        full_gpd.crs = {'init' :'epsg:4326'}

        return(full_gpd)

    def initialReadIn(self, fpath=None, wktField='Wkt'):
        """
        Convert the OSM object to a networkX object

        :param fpath: path to CSV file with roads to read in
        :param wktField: wktField name
        :returns: Networkx Multi-digraph
        """
        if isinstance(fpath, str):
            edges_1 = pd.read_csv(fpath)
            edges_1 = edges_1[wktField].apply(lambda x: loads(x))
        elif isinstance(fpath, gpd.GeoDataFrame):
            edges_1 = fpath
        else:
            try:
                edges_1 = self.roadsGPD
            except:
                self.generateRoadsGDF()
                edges_1 = self.roadsGPD

        edges = edges_1.copy()
        node_bunch = list(set(list(edges['u']) + list(edges['v'])))

        def convert(x):
            u = x.u
            v = x.v
            data = {'Wkt':x.Wkt,
                   'id':x.id,
                   'infra_type':x.infra_type,
                   'one_way':x.one_way,
                   'osm_id':x.osm_id,
                   'key': x.key,
                   'length':x.length}
            return (u, v, data)

        edge_bunch = edges.apply(lambda x: convert(x), axis = 1).tolist()

        G = nx.MultiDiGraph()
        G.add_nodes_from(node_bunch)
        G.add_edges_from(edge_bunch)

        for u, data in G.nodes(data = True):
            if type(u) == str:
                q = tuple(float(x) for x in u[1:-1].split(','))
            if type(u) == tuple:
                q = u
            data['x'] = q[0]
            data['y'] = q[1]
        G = nx.convert_node_labels_to_integers(G)
        self.network = G

        return G
