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
from geopy.distance import vincenty
from boltons.iterutils import pairwise
from shapely.wkt import loads,dumps

import osmnx as ox


def generate_traffic_metrics(self, *filenames):
        """
        Takes in one or more Mapbox traffic files in csv form merges them if needed and calculates metrics such as min, max, and mean speed columns

        :param *filenames: one or more Mapbox traffic files in csv form. 
        :returns: the new dataframe with metrics into the traffic_simplified_df DataFrame property of the OSM_to_network object
        """

        print(len(filenames))
        print(filenames[0])

        final_df = pd.read_csv(filenames[0], header = None)
        print(f"finished reading {filenames[0]} into dataframe")
        #print(len(final_df))

        if len(filenames) > 1:
          for filename in filenames[1:]:
              print('FILE', filename)
              base = os.path.basename(filename)
              print(os.path.splitext(base)[0] + "_df")
              new_filename = os.path.splitext(base)[0] + "_df"

              df_to_merge = pd.read_csv(filename, header = None)
              #print(len(df_to_merge))

              final_df = final_df.append(df_to_merge, ignore_index = True, verify_integrity = True)
              print(f"finished merging {new_filename} into combined dataframe")
              #print(len(final_df))

        print("calculating min, max, and mean values.")
        def get_speeds(x):
            ''' Return Min, Max, and Mean speed '''
            x_vals = x[2:]
            return([min(x_vals), max(x_vals), np.mean(x_vals)]) #, np.argmax(x_vals)
    
        traffic_vals = final_df.apply(lambda x: get_speeds(x), axis = 1, result_type = "expand")
        traffic_vals.columns = ['min_speed','max_speed','mean_speed']

        traffic_simplified = final_df.loc[:,[0,1]]
        traffic_simplified.columns = ['FROM_NODE', "TO_NODE"]
        traffic_simplified = traffic_simplified.join(traffic_vals)

        print("finished calculating min, max, and mean values. Printing traffic_simplified head")
        print(traffic_simplified.head())

        return traffic_simplified


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

    def __init__(self, osmFile, traffic_simplified_df, includeFerries = False):
        """
        Generate a networkX object from a osm file
        """
        self.osmFile = osmFile
        #self.roads_raw = self.fetch_roads_and_ferries(osmFile) if includeFerries else self.fetch_roads(osmFile)
        #self.traffic_simplified_df = self.generate_traffic_metrics(*filenames)

        self.network = self.fetch_roads_w_traffic(osmFile, traffic_simplified_df)
        #fetch_roads_list = self.fetch_roads_w_traffic(osmFile, traffic_simplified_df)
        #self.roads_raw = fetch_roads_list[0]
        #self.nodes_raw = fetch_roads_list[1]

        # next step is to do network clean
        # look into if it is easier to do this with GDF inputs instead of graph
        # also look if easier to apply Mapbox speeds during the clean graph step


    def apply_traffic_speeds_to_roads_raw(self, *filenames):
        """
        apply_traffic_speeds_to_roads_raw

        :param *filenames: one or more Mapbox traffic files in csv form. 
        :returns: None - Mapbox traffic speeds get added to edges in roads_raw
        """

        in_df = self.roads_raw

        # if the dataframe already has min_speed, max_speed, and mean speed columns, these should be deleted
        cols = [c for c in in_df.columns if c.lower()[-6:] != '_speed']
        in_df = in_df[cols]

        #print("print in_df")
        #print(in_df)

        print(len(filenames))
        print(filenames[0])

        final_df = pd.read_csv(filenames[0], header = None)
        print(f"finished reading {filenames[0]} into dataframe")
        #print(len(final_df))

        if len(filenames) > 1:
          for filename in filenames[1:]:
              print('FILE', filename)
              base = os.path.basename(filename)
              print(os.path.splitext(base)[0] + "_df")
              new_filename = os.path.splitext(base)[0] + "_df"

              df_to_merge = pd.read_csv(filename, header = None)
              #print(len(df_to_merge))

              final_df = final_df.append(df_to_merge, ignore_index = True, verify_integrity = True)
              print(f"finished merging {new_filename} into combined dataframe")
              #print(len(final_df))

        print("calculating min, max, and mean values.")
        def get_speeds(x):
            ''' Return Min, Max, and Mean speed '''
            x_vals = x[2:]
            return([min(x_vals), max(x_vals), np.mean(x_vals)]) #, np.argmax(x_vals)
    
        traffic_vals = final_df.apply(lambda x: get_speeds(x), axis = 1, result_type = "expand")
        traffic_vals.columns = ['min_speed','max_speed','mean_speed']

        traffic_simplified = final_df.loc[:,[0,1]]
        traffic_simplified.columns = ['FROM_NODE', "TO_NODE"]
        traffic_simplified = traffic_simplified.join(traffic_vals)

        print("finished calculating min, max, and mean values. Printing traffic_simplified head")
        print(traffic_simplified.head())

        print("adding the traffic speeds to the edges")
        # doing a left join so that all of the original edges remain
        roads_raw_w_traffic = in_df.merge(traffic_simplified, how = "left", left_on = ['stnode','endnode'], right_on = ['FROM_NODE','TO_NODE'])

        #print("print roads_raw_w_traffic")
        #print(roads_raw_w_traffic)

        # calculate the percentage of roads that have a mapbox speed
        # count the number of rows that have a Mapbox speed of 0 or more
        seriesObj = roads_raw_w_traffic.apply(lambda x: True if x['mean_speed'] > 0 else False, axis = 1)
        numOfRows = len(seriesObj[seriesObj == True].index)
        print('{:.1%} of roads have a traffic speed'.format(numOfRows/len(roads_raw_w_traffic)))

        self.roads_raw = roads_raw_w_traffic


    def generateRoadsGDF(self, in_df = None, outFile='', verbose = False):
        """
        post-process roads GeoDataFrame adding additional attributes

        :param in_df: Optional input GeoDataFrame
        :param outFile: optional parameter to output a csv with the processed roads
        :returns: Length of line in kilometers
        """
        if type(in_df) != gpd.geodataframe.GeoDataFrame:
            in_df_roads_raw = self.roads_raw
            in_df_nodes_raw = self.nodes_raw

        # get all intersections, 

        # start_time = time.time()
        # get_all_intersections_and_create_nodes_list = self.get_all_intersections_and_create_nodes(in_df_roads_raw, in_df_nodes_raw, unique_id = 'osm_id', verboseness = verbose)
        # roads = get_all_intersections_and_create_nodes_list[0]
        # nodes = get_all_intersections_and_create_nodes_list[1]
        # end_time = time.time()
        # print(end_time - start_time)

        # reset index
        in_df_nodes_raw = in_df_nodes_raw.reset_index(drop=True)

        #roads = self.get_all_intersections_and_create_nodes(in_df_roads_raw, in_df_nodes_raw, unique_id = 'osm_id', verboseness = verbose)

        # add new key column that has a unique id
        #roads['key'] = ['edge_'+str(x+1) for x in range(len(roads))]
        #np.arange(1,len(roads)+1,1)

        #def get_nodes(x):
        #    return list(x.geometry.coords)[0],list(x.geometry.coords)[-1]

        # generate all of the nodes per edge and to and from node columns
        #nodes = gpd.GeoDataFrame(roads.apply(lambda x: get_nodes(x),axis=1).apply(pd.Series))
        #nodes.columns = ['u','v']

        #roads = pd.concat([roads,nodes],axis=1)

        # compute the length per edge
        in_df_roads_raw['length'] = in_df_roads_raw.geometry.apply(lambda x : self.line_length(x))
        #roads.rename(columns = {'geometry':'Wkt'}, inplace = True)

        if outFile != '':
            roads.to_csv(outFile)

        self.roadsGDF = in_df_roads_raw
        self.nodesGDF = in_df_nodes_raw

    def filterRoads(self, acceptedRoads = ['primary','primary_link','secondary','secondary_link','motorway','motorway_link','trunk','trunk_link']):
        """
        Extract certain times of roads from the OSM before the netowrkX conversion 

        :param acceptedRoads: [ optional ] acceptedRoads [ list of strings ] 
        :returns: None - the raw roads are filtered based on the list of accepted roads
        """

        self.roads_raw = self.roads_raw.loc[self.roads_raw.infra_type.isin(acceptedRoads)]
        
    def add_complete_edges(self,G, all_edges):
        # the list of values OSM uses in its 'oneway' tag to denote True
        # https://www.geofabrik.de/de/data/geofabrik-osm-gis-standard-0.7.pdf
        osm_oneway_values = ["yes", "true", "1", "-1", "T", "F"]
        
        for edge in all_edges:
            if ("oneway" in edge[2] and edge[2]["oneway"] in osm_oneway_values):
                if edge[2]["oneway"] == "-1" or edge[2]["oneway"] == "T":
                    # paths with a one-way value of -1 or T are one-way, but in the
                    # reverse direction of the nodes' order, see osm documentation
                    #not tested
                    print('a reverse one way')
                    #data["nodes"] = list(reversed(data["nodes"]))
                    edge[2]["oneway"] = "one_way"
                    G.add_edges_from([(edge[1],edge[0],edge[2])])
                else:
                    # add this path (in only one direction) to the graph
                    #print('a one way edge')
                    edge[2]["oneway"] = "one_way"
                    G.add_edges_from([(edge[0],edge[1],edge[2])])

            #add later
            elif ("junction" in edge[2] and edge[2]["junction"] == "roundabout"):
                #roundabout are also oneway but not tagged as is
                #_add_path(G, data, one_way=True)
                edge[2]["oneway"] = "one_way"
                G.add_edges_from([(edge[0],edge[1],edge[2])])

            # else, this path is not tagged as one-way or it is a walking network
            # (you can walk both directions on a one-way street)
            else:
                # add this path (in both directions) to the graph and set its
                # 'oneway' attribute to False. if this is a walking network, this
                # may very well be a one-way street (as cars/bikes go), but in a
                # walking-only network it is a bi-directional edge
                #print('adding a reverse edge')
                #print([edge[1],edge[0],edge[2]])
                #all_edges.append([edge[1],edge[0],edge[2]])
                G.add_edges_from([(edge[1],edge[0],edge[2])])
                G.add_edges_from([(edge[0],edge[1],edge[2])])
        
        print('done pre-processing edges')
        #G.add_edges_from(all_edges)

        return G

    def fetch_roads_w_traffic(self, data_path, traffic_simplified_df):
        import osmium, logging
        wkbfab = osmium.geom.WKBFactory()
        import shapely.wkb as wkblib

        # extract highways
        class HighwayExtractor(osmium.SimpleHandler):
            def __init__(self):
                osmium.SimpleHandler.__init__(self) 
                self.nodes = []
                #self.raw_h = []
                self.highways = []
                self.broken_highways = []
                self.total = 0
                self.num_nodes = 0
            
            def way(self, w):
                #self.raw_h.append(w)
                try:
                    nodes = [x.ref for x in w.nodes]
                    wkb = wkbfab.create_linestring(w)
                    shp = wkblib.loads(wkb, hex=True)
                    if 'highway' in w.tags:
                        #info = [w.id, nodes, shp, w.tags['highway']]
                        info = {'osmid':w.id, 'nodes':nodes, 'shp':shp, 'highway':w.tags['highway']}
                        if 'maxspeed' in w.tags:
                            #info.append(w.tags['maxspeed'])
                            info['maxspeed']=w.tags['maxspeed']
                        if 'oneway' in w.tags:
                            #info.append(w.tags['oneway'])
                            info['oneway']=w.tags['oneway']
                        if 'junction' in w.tags:
                            info['junction']=w.tags['junction']
                        self.highways.append(info)
                except:
                    print('hit exception')
                    try:
                        nodes = [x for x in w.nodes if x.location.valid()]
                        if len(nodes) > 1:
                            shp = LineString([Point(x.location.x, x.location.y) for x in nodes])
                            info = {'osmid':w.id, 'nodes':nodes, 'shp':shp, 'highway':w.tags['highway']}
                            if 'maxspeed' in w.tags:
                                info['maxspeed']=w.tags['maxspeed']
                            if 'oneway' in w.tags:
                                info['oneway']=w.tags['oneway']
                            if 'junction' in w.tags:
                                info['junction']=w.tags['junction']
                            self.highways.append(info)
                        else:
                            self.broken_highways.append(w.id)
                    except:
                        print("way %s may not have nodes" % w.id)
                    logging.warning("Error Processing OSM Way %s" % w.id)

        h = HighwayExtractor()
        h.apply_file(data_path, locations = True)

        print('finished with Osmium data extraction')
        print(len(h.highways))
        print(len(h.broken_highways))

        all_nodes = []
        all_edges = []

        for x in h.highways:
            #print('looping 1')
            #print(f'print count: {count}')
            highway_node_list = []
            highway_edge_list = []
            traffic_node_matches = 0
            #if 'oneway' in x.keys():
                #print('print x["oneway"]')
                #print(x['oneway'])
            for n_idx in range(0, (len(x['nodes']) - 1)):
                #print(f'print n_idx: {n_idx}')
                try:
                    osm_id_from = x['nodes'][n_idx].ref
                except:
                    osm_id_from = x['nodes'][n_idx]
                try:
                    osm_id_to   = x['nodes'][n_idx+1].ref
                except:
                    osm_id_to   = x['nodes'][n_idx+1]
                try:
                    osm_coords_from = list(x['shp'].coords)[n_idx]
                    osm_coords_to = list(x['shp'].coords)[n_idx+1]

                    #print(f'print osm_id_from: {osm_id_from}')
                    if len(traffic_simplified_df.loc[(traffic_simplified_df.FROM_NODE == osm_id_from) | (traffic_simplified_df.TO_NODE == osm_id_from)]):
                      #print('hit')
                      traffic_node_matches += 1
                    #print(osm_coords_from[0])
                    #create a node
                    #all_nodes.append([osm_id_from, { 'x' : osm_coords_from[0], 'y' : osm_coords_from[1] }])
                    #highway_node_list.append([osm_id_from, Point(osm_coords_from[0], osm_coords_from[1])])
                    highway_node_list.append([osm_id_from, {'x': osm_coords_from[0], 'y': osm_coords_from[1]}])
                    #print(n_idx)
                    #print(len(x[1]) - 1)
                    if n_idx == (len(x['nodes']) - 2):
                        #print('last element')
                        #create a node
                        #print(osm_coords_to)
                        #all_nodes.append([osm_id_to, { 'x' : osm_coords_to[0], 'y' : osm_coords_to[1]} ])
                        #highway_node_list.append([osm_id_to, Point(osm_coords_to[0], osm_coords_to[1])])
                        highway_node_list.append([osm_id_to, {'x': osm_coords_to[0], 'y': osm_coords_to[1]}])
                    edge = LineString([osm_coords_from, osm_coords_to])
                    #attr = {'osmid':x['osmid'], 'infra_type':x['highway'], 'geometry':edge}
                    attr = {'osmid':x['osmid'], 'infra_type':x['highway']}
                    # if maxspeed tag
                    if 'maxspeed' in x.keys():
                        attr['maxspeed'] = x['maxspeed']
                    # if oneway tag
                    if 'oneway' in x.keys():
                        attr['oneway'] = x['oneway']
                    # if oneway tag
                    if 'junction' in x.keys():
                        attr['junction'] = x['junction']
                    #Create an edge from the list of nodes in both directions
                    #print(f'adding edge with {osm_id_from}')
                    highway_edge_list.append([osm_id_from, osm_id_to, attr])
                    #all_edges.append([osm_id_to, osm_id_from, attr])
                except:
                    #logging.warning(f"Error adding edge between nodes {osm_id_from} and {osm_id_to}")
                    print(f"Error adding edge between nodes {osm_id_from} and {osm_id_to}")

            if traffic_node_matches > 0:
                #print(f'traffic node matches: {traffic_node_matches}')
                for node_item in highway_node_list:
                    all_nodes.append(node_item)
                # add seperate edges
                for hwy in highway_edge_list:
                    if len(traffic_simplified_df.loc[(traffic_simplified_df.FROM_NODE == hwy[0]) & (traffic_simplified_df.TO_NODE == hwy[1])]) > 0:
                        traffic_match = traffic_simplified_df.loc[(traffic_simplified_df.FROM_NODE == hwy[0]) & (traffic_simplified_df.TO_NODE == hwy[1])].iloc[0]
                        #print(f'print traffic_match: {traffic_match}')
                        #print(f'print hwy[2]: {hwy[2]}')
                        hwy[2]['traffic_min_speed'] = traffic_match['min_speed']
                        hwy[2]['traffic_max_speed'] = traffic_match['max_speed']
                        hwy[2]['traffic_mean_speed'] = traffic_match['mean_speed']
                        all_edges.append(hwy)
                    elif len(traffic_simplified_df.loc[(traffic_simplified_df.FROM_NODE == hwy[1]) & (traffic_simplified_df.TO_NODE == hwy[0])]) > 0:
                        traffic_match = traffic_simplified_df.loc[(traffic_simplified_df.FROM_NODE == hwy[1]) & (traffic_simplified_df.TO_NODE == hwy[0])].iloc[0]
                        hwy[2]['traffic_min_speed'] = traffic_match['min_speed']
                        hwy[2]['traffic_max_speed'] = traffic_match['max_speed']
                        hwy[2]['traffic_mean_speed'] = traffic_match['mean_speed']
                        all_edges.append(hwy)
                    else:
                        all_edges.append(hwy)
            else:
                for node_item in highway_node_list:
                    all_nodes.append(node_item)
                # add complete edge and only 1st and end node
                #print(f'print 1st hwy node: {highway_node_list[0]}')
                #all_nodes.append(highway_node_list[0])
                #print(f'print last hwy node: {highway_node_list[-1]}')
                #all_nodes.append(highway_node_list[-1])
                #all_edges.append()
                #print('print hwy')
                #print(x[2])
                #all_edges.append([highway_node_list[0][0], highway_node_list[-1][0], {'osm_id':x[0], 'infra_type':x[3], 'geometry':x[2]}])
                for hwy in highway_edge_list:
                    all_edges.append(hwy)

        print('finished building node edge lists')
        print('all_edges length')
        print(len(all_edges))

        #print('print all_nodes')
        #print(all_nodes)

        #all_nodes_pd = pd.DataFrame(all_nodes, columns = ['osm_id', 'geometry'])

        # it may be possible to get duplicate nodes if two lines share the same node, therefore remove duplicates but keep the first occurance
        #all_nodes_pd.drop_duplicates(subset = "osm_id", keep = 'first', inplace = True) 
        #all_nodes_gdf = gpd.GeoDataFrame(all_nodes_pd, geometry = 'geometry')

        # flatten all_edges to list
        #for edge in all_edges:
          #print(edge[2])
          #for k, v in edge[2].items():
              #print(v)
              #edge.append(v) 
          #edge.pop(2)
          #print(edge)

        #all_edges_df = pd.DataFrame(all_edges, columns = ['stnode', 'endnode', 'osm_id', 'infra_type', 'geometry', 'min_speed', 'max_speed', 'mean_speed'])
        #all_edges_gdf = gpd.GeoDataFrame(all_edges_df, geometry = 'geometry')

        #print('finished building node and edge GeoDataFrames')
        #print('all_edges_gdf length')
        #print(len(all_edges_gdf))
        
        G = nx.MultiDiGraph(crs='epsg:4326')
        # add each osm node to the graph
        print('adding nodes')
        G.add_nodes_from(all_nodes)
        print('adding edges')
        # add each osm way (ie, a path of edges) to the graph
        G = self.add_complete_edges(G, all_edges)
        

        # add length (great circle distance between nodes) attribute to each edge to
        # use as weight
        if len(G.edges) > 0:
            G = ox.utils_graph.add_edge_lengths(G)

        #return [all_edges_gdf, all_nodes_gdf]
        return G

    def fetch_roads(self, data_path):
        """
        Extracts roads from an OSM PBF

        :param data_path: The directory of the shapefiles consisting of edges and nodes
        :returns: a road GeoDataFrame
        """

        if data_path.split('.')[-1] == 'pbf':
            driver = ogr.GetDriverByName("OSM")
            data = driver.Open(data_path)
            sql_lyr = data.ExecuteSQL("SELECT osm_id,highway FROM lines WHERE highway IS NOT NULL")
            roads = []

            for feature in sql_lyr:
                if feature.GetField("highway") is not None:
                    osm_id = feature.GetField("osm_id")
                    shapely_geo = loads(feature.geometry().ExportToWkt())
                    if shapely_geo is None:
                        continue
                    highway = feature.GetField("highway")
                    roads.append([osm_id,highway,shapely_geo])

            if len(roads) > 0:
                road_gdf = gpd.GeoDataFrame(roads,columns=['osm_id','infra_type','geometry'],crs={'init': 'epsg:4326'})
                return road_gdf

        elif data_path.split('.')[-1] == 'shp':
            road_gdf = gpd.read_file(data_path)
            return road_gdf

        else:
            print('No roads found')

    # def fetch_roads_and_ferries(self, data_path):
    #     """
    #     Extracts roads and ferries from an OSM PBF

    #     :param data_path: The directory of the shapefiles consisting of edges and nodes
    #     :returns: a road GeoDataFrame
    #     """

    #     if data_path.split('.')[-1] == 'pbf':

    #         driver = ogr.GetDriverByName('OSM')
    #         data = driver.Open(data_path)
    #         sql_lyr = data.ExecuteSQL("SELECT * FROM lines")

    #         roads=[]

    #         for feature in sql_lyr:
    #             if feature.GetField('man_made'):
    #                 if "pier" in feature.GetField('man_made'):
    #                     osm_id = feature.GetField('osm_id')
    #                     shapely_geo = loads(feature.geometry().ExportToWkt())
    #                     if shapely_geo is None:
    #                         continue
    #                     highway = 'pier'
    #                     roads.append([osm_id,highway,shapely_geo])
    #             elif feature.GetField('other_tags'):
    #                 if "ferry" in feature.GetField('other_tags'):
    #                     osm_id = feature.GetField('osm_id')
    #                     shapely_geo = loads(feature.geometry().ExportToWkt())
    #                     if shapely_geo is None:
    #                         continue
    #                     highway = 'ferry'
    #                     roads.append([osm_id,highway,shapely_geo])
    #                 elif feature.GetField('highway') is not None:
    #                     osm_id = feature.GetField('osm_id')
    #                     shapely_geo = loads(feature.geometry().ExportToWkt())
    #                     if shapely_geo is None:
    #                         continue
    #                     highway = feature.GetField('highway')
    #                     roads.append([osm_id,highway,shapely_geo])
    #             elif feature.GetField('highway') is not None:
    #                 osm_id = feature.GetField('osm_id')
    #                 shapely_geo = loads(feature.geometry().ExportToWkt())
    #                 if shapely_geo is None:
    #                     continue
    #                 highway = feature.GetField('highway')
    #                 roads.append([osm_id,highway,shapely_geo])

    #         if len(roads) > 0:
    #             road_gdf = gpd.GeoDataFrame(roads,columns=['osm_id','infra_type','geometry'],crs={'init': 'epsg:4326'})
    #             return road_gdf

    #     elif data_path.split('.')[-1] == 'shp':
    #         road_gdf = gpd.read_file(data_path)
    #         return road_gdf

    #     else:
    #         print('No roads found')

    def line_length(self, line, ellipsoid = 'WGS-84'):
        """
        Returns length of a line in kilometers, given in geographic coordinates. Adapted from https://gis.stackexchange.com/questions/4022/looking-for-a-pythonic-way-to-calculate-the-length-of-a-wkt-linestring#answer-115285

        :param line: a shapely LineString object with WGS-84 coordinates
        :param string ellipsoid: string name of an ellipsoid that `geopy` understands (see http://geopy.readthedocs.io/en/latest/#module-geopy.distance)
        :returns: Length of line in kilometers
        """
        
        if line.geometryType() == 'MultiLineString':
            return sum(line_length(segment) for segment in line)

        return sum(
                    vincenty(tuple(reversed(a)), tuple(reversed(b)), ellipsoid=ellipsoid).kilometers
                    for a, b in pairwise(line.coords)
        )

    def get_all_intersections_and_create_nodes(self, in_df_roads_raw, in_df_nodes_raw, idx_osm = None, unique_id = 'osm_id', verboseness = False):
        """
        Processes GeoDataFrame and splits edges as intersections

        :param shape_input: Input GeoDataFrame
        :param idx_osm: The geometry index name
        :param unique_id: The unique id field name
        :returns: returns processed GeoDataFrame
        """

        # Initialize Rtree
        idx_inters = index.Index()

        if idx_osm is None:
            idx_osm = in_df_roads_raw['geometry'].sindex

        # Find all the intersecting lines to prepare for cutting
        count = 0
        tLength = in_df_roads_raw.shape[0]
        inters_done = {}
        new_lines = []
        allCounts = []

        hits_0 = 0
        hits_1 = 0
        hits_2 = 0
        hits_3 = 0
        hits_4_or_more = 0

        print("length of in_df_roads_raw")
        print(len(in_df_roads_raw))

        for idx, row in in_df_roads_raw.iterrows():
            #print(row)
            key1 = row[f'{unique_id}']
            stnode = row.stnode
            endnode = row.endnode
            infra_type = row.infra_type
            line = row.geometry
            min_speed = row.min_speed
            max_speed = row.max_speed
            mean_speed = row.mean_speed
            if count % 10000 == 0 and verboseness == True:
                print("Processing %s of %s" % (count, tLength))
            count += 1
            intersections = in_df_roads_raw.iloc[list(idx_osm.intersection(line.bounds))]
            intersections = dict(zip(list(intersections[f'{unique_id}']),list(intersections.geometry)))
            # ignore self-intersecting lines
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
                    if "Point" == inter.type:
                        idx_inters.insert(0, inter.bounds, inter)
                        # print('print Point idx_inters')
                        # print(inter.bounds)
                        # ex: (79.9271893, 7.0261838, 79.9271893, 7.0261838)
                        # print(inter)
                        # ex: POINT (79.92718929999999 7.0261838)
                    elif "MultiPoint" == inter.type:
                        for pt in inter:
                            idx_inters.insert(0, pt.bounds, pt)

            #  rtree.index.Index.intersection() will return you index entries that cross or are contained within the given query window
            hits = [n.object for n in idx_inters.intersection(line.bounds, objects = True)]

            # print('hits length')
            # print(len(hits))

            if len(hits) != 0:
                if len(hits) == 1:
                    hits_1 += 1
                if len(hits) == 2:
                    hits_2 += 1
                if len(hits) == 3:
                    hits_3 += 1
                if len(hits) >= 4:
                    hits_4_or_more += 1
                    # print('hits length')
                    # print(len(hits))
                #try:
                # cut lines where necessary and save all new linestrings to a list
                out = shapely.ops.split(line, MultiPoint(hits))
                #if len(out) > 1:
                    # print(r"The length of out is: ")
                    # print(len(out))
                    # print(r"The length of hits is: ")
                    # print(len(hits))
                    # print(r"length of new_lines before append is: ")
                    # print(len(new_lines))
                #new_lines.append([{'stnode': stnode, 'endnode': endnode, 'osm_id': key1, 'infra_type': infra_type, 'geometry': LineString(x)} for x in out.geoms])
                first = True
                out_length = len(out)
                loop_counter = 1
                for num, x in enumerate(out.geoms, start=1):
                    if not first:
                        #print('not the first entry')
                        pass
                    if first:
                        first = False
                        if num == out_length:
                            # if last item
                            # in this case there is only out geometry, this case occurs often when a hit occurs within the bounding box but not on the line itself
                            # therefore we just need to append the line without any cuts
                            # print("appending single line without any cuts")
                            new_lines.append([{'stnode': stnode, 'endnode': endnode, 'osm_id': key1, 'infra_type': infra_type, 'geometry': LineString(x), 'min_speed': min_speed, 'max_speed': max_speed, 'mean_speed': mean_speed}])
                        else:
                            #print(f"print loop_counter (starts at 1) statement 1: {loop_counter}")
                            # a new node will need to be created with a new node_id
                            #print(f"print stnode: {stnode}")
                            #print(f"print endnode: {endnode}")

                            # This can produce a buffer overflow if it is too large, therefore trim each part to 5 characters before concatenating
                            new_node_id = np.int64(str(99) + str(stnode)[4:] + str(endnode)[:4])
                            # print("print LineString(x)")
                            # print(LineString(x))

                            #print("created new node")

                            # nodes of split line
                            u = list(LineString(x).coords)[0]
                            v = list(LineString(x).coords)[-1]

                            # round x coord to 6 decimals and y coord to 7 decimals
                            #rounded_v = (round(v[0], 6),round(v[1], 7))

                            #print("assigned u and v")
                            #print(f'print u: {u} and v: {v}')

                            #node_geom = Point(rounded_v[0], rounded_v[1])
                            node_geom = Point(v[0], v[1])

                            # print("assigned node_geom")
                            # print("print node_geom")
                            # print(node_geom)
                            # print(list(node_geom.coords))

                            mini_gdf = gpd.GeoDataFrame({'osm_id': new_node_id, 'geometry': [node_geom]}, crs = 'epsg:4326')

                            #print(f"appending node with osm_id: {new_node_id} and geometry: {node_geom}")
                            
                            #print('print in_df_nodes_raw length before')
                            #print(len(in_df_nodes_raw))
                            in_df_nodes_raw = in_df_nodes_raw.append(mini_gdf)
                            #print('print in_df_nodes_raw length after')
                            #print(len(in_df_nodes_raw))

                            #print(f"appending line with stnode: {stnode}, endnode: {new_node_id}, and osm_id: {key1}")
                            new_lines.append([{'stnode': stnode, 'endnode': new_node_id, 'osm_id': key1, 'infra_type': infra_type, 'geometry': LineString(x), 'min_speed': min_speed, 'max_speed': max_speed, 'mean_speed': mean_speed}])

                            stnode = new_node_id
                    else:
                        #print('print num')
                        #print(num)
                        #print(f"print loop_counter (starts at 1)  statement 2: {loop_counter}")

                        if num == out_length:
                            # if last item

                            #print(f"appending line with stnode: {stnode}, endnode: {endnode}, and osm_id: {key1}")
                            new_lines.append([{'stnode': stnode, 'endnode': endnode, 'osm_id': key1, 'infra_type': infra_type, 'geometry': LineString(x), 'min_speed': min_speed, 'max_speed': max_speed, 'mean_speed': mean_speed}])

                        else:
                            # a new node will need to be created with a new node_id
                            new_node_id = new_node_id + 1

                            # print("print LineString(x)")
                            # print(LineString(x))

                            # nodes of split line
                            u = list(LineString(x).coords)[0]
                            v = list(LineString(x).coords)[-1]

                            # round x coord to 6 decimals and y coord to 7 decimals
                            #rounded_v = (round(v[0], 6),round(v[1], 7))

                            #print("assigned u and v")
                            #print(f'print u: {u} and v: {v}')

                            #node_geom = Point(rounded_v[0], rounded_v[1])
                            node_geom = Point(v[0], v[1])

                            mini_gdf = gpd.GeoDataFrame({'osm_id': new_node_id, 'geometry':[node_geom] }, crs = 'epsg:4326')

                            #print(f"appending node with osm_id: {new_node_id} and geometry: {node_geom}")
                            in_df_nodes_raw = in_df_nodes_raw.append(mini_gdf)

                            #print(f"appending line with stnode: {stnode}, endnode: {new_node_id}, and osm_id: {key1}")
                            new_lines.append([{'stnode': stnode, 'endnode': new_node_id, 'osm_id': key1, 'infra_type': infra_type, 'geometry': LineString(x), 'min_speed': min_speed, 'max_speed': max_speed, 'mean_speed': mean_speed}])

                            stnode = new_node_id
                    loop_counter += 1
                # if len(out) > 1:
                #     #print("length of new_lines after append is: ")
                #     #print(len(new_lines))
                # except:
                #     pass
            else:
                hits_0 += 1
                #new_lines.append([{'geometry': line, 'osm_id': key1, 'infra_type': infra_type}])
                new_lines.append([{'stnode': stnode, 'endnode': endnode, 'osm_id': key1, 'infra_type': infra_type, 'geometry': line, 'min_speed': min_speed, 'max_speed': max_speed, 'mean_speed': mean_speed}])

        # Create one big list and treat all the cutted lines as unique lines
        flat_list = []
        #all_data = {}

        print('new_lines count')
        print(len(new_lines))

        # item for sublist in new_lines for item in sublist
        i = 1
        for sublist in new_lines:
            if sublist is not None:
                for dict_inside_list in sublist:
                    dict_inside_list['id'] = i
                    flat_list.append(dict_inside_list)
                    i += 1
                    #all_data[i] = item

        print('flat_list count')
        print(len(flat_list))

        print('print 1st 3 items of flat_list')
        print(flat_list[:3])

        print('print hits 0 count')
        print(hits_0)

        print('print hits 1 count')
        print(hits_1)

        print('print hits 2 count')
        print(hits_2)

        print('print hits 3 count')
        print(hits_3)

        print('print hits 4 or more count')
        print(hits_4_or_more)

        print('print in_df_nodes_raw length')
        print(len(in_df_nodes_raw))

        # Transform into geodataframe and add coordinate system
        in_df_roads_raw = gpd.GeoDataFrame(flat_list, geometry ='geometry')
        in_df_roads_raw.crs = {'init' :'epsg:4326'}

        return [in_df_roads_raw, in_df_nodes_raw]

    def initialReadIn(self, fpath = None, wktField = 'Wkt'):
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
                edges_1 = self.roadsGDF
                nodes_1 = self.nodesGDF
            except:
                self.generateRoadsGDF()
                edges_1 = self.roadsGDF
                nodes_1 = self.nodesGDF

        edges = edges_1.copy()
        nodes = nodes_1.copy()
        #node_bunch = list(set(list(edges['u']) + list(edges['v'])))
        
        def convert(x):
            u = x.stnode
            v = x.endnode
            data = {
                   'osm_id':x.osm_id,
                   'geometry':x.geometry,
                   'infra_type':x.infra_type,
                   'min_speed':x.min_speed,
                   'max_speed':x.max_speed,
                   'mean_speed':x.mean_speed,
                   #'key': x.key,
                   'length':x.length}
            return (u, v, data)

        def convert_reflected(x):
            u = x.stnode
            v = x.endnode
            data = {
                   'osm_id':x.osm_id,
                   'geometry':x.geometry,
                   'infra_type':x.infra_type,
                   'min_speed':x.min_speed,
                   'max_speed':x.max_speed,
                   'mean_speed':x.mean_speed,
                   #'key': x.key,
                   'length':x.length}
            return (v, u, data)

        edge_bunch = edges.apply(lambda x: convert(x), axis = 1).tolist()

        # now add reflected edges
        edge_bunch_reflected = edges.apply(lambda x: convert_reflected(x), axis = 1).tolist()

        def convert_node(x):
            osm_id = x.osm_id
            data = {'geometry': x.geometry}
            return (osm_id, data)

        node_bunch = nodes.apply(lambda x: convert_node(x), axis = 1).tolist()

        print('print edge bunch')
        print(edge_bunch[:10])

        print('print node bunch')
        print(node_bunch[:10])

        G = nx.MultiDiGraph()
        #G.add_nodes_from(node_bunch)
        G.add_nodes_from(node_bunch)
        G.add_edges_from(edge_bunch)
        G.add_edges_from(edge_bunch_reflected)

        #print('print edges in Multi-digraph')
        #print(G.edges)

        # print('print nodes in Multi-digraph')
        # print(G.nodes)

        # is it really necessary to have seperate 'x' and 'y' columns?
        # for u, data in G.nodes(data = True):
        #     if type(u) == str:
        #         q = tuple(float(x) for x in u[1:-1].split(','))
        #     if type(u) == tuple:
        #         q = u
        #     data['x'] = q[0]
        #     data['y'] = q[1]

        # Returns a copy of the graph G with the nodes relabeled using consecutive integers
        #G = nx.convert_node_labels_to_integers(G)
        self.network = G
        
        return G
