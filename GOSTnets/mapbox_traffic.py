import os, sys, time, importlib, logging

import geopandas as gpd
import pandas as pd
import networkx as nx
import numpy as np
import shapely.wkb as wkblib

from shapely.geometry import Point, LineString

def simplify_mapbox_csv(csv_file, simplification="basic"):
    """ Generate a simplified csv_file from mapbox traffic. 
        https://docs.mapbox.com/traffic-data/overview/data/#typical-data-file
    
    :param csv_file:
        path to csv file containing mapbox traffic information
    :param simplification:
        string defining the simplification method. Options are ["basic"]
    """
    
    traffic = pd.read_csv(csv_file, header=None)
    def get_speeds(x):
        ''' Return Min, Max, and Mean speed '''
        x_vals = x[2:]
        return([min(x_vals), max(x_vals), np.mean(x_vals)]) #, np.argmax(x_vals)
        
    traffic_vals = traffic.apply(lambda x: get_speeds(x), axis=1, result_type="expand")
    traffic_vals.columns = ['min_speed','max_speed','mean_speed']
    traffic_simplified = traffic.loc[:,[0,1]]
    traffic_simplified.columns = ['FROM_NODE', "TO_NODE"]
    traffic_simplified = traffic_simplified.join(traffic_vals)
    return(traffic_simplified)
    
def attach_traffic_data(G, traffic_data, 
                        left_on_fields = ['stnode','endnode'],
                        right_on_fields = ['FROM_NODE','TO_NODE'],
                        length_field = "length",
                        speed_field = "min_speed"):
    """ Attach the traffic information from the traffic csv to the G networkx object
    
    :param G:
        networkX multiDiGraph
    :param traffic_data:
        Geopandas data frame created from function simplify_mapbox_csv
    :returns G:
        networkX multiDiGraph
    """
    edges_gdf = edge_gdf_from_graph(G)
    nodes_gdf = node_gdf_from_graph(G)
    ### TODO: right now all edges in the attributed nodes dataset are included in the speed dataset
    #   There needs to be a step to provide a default value based on infra_value if the edges
    #   are not properly attributed
    attributed_nodes = edges_gdf.merge(traffic_simplified, left_on=left_on_fields, right_on=right_on_fields)
    attributed_nodes['time'] = attributed_nodes[length_field] * attributed_nodes[speed_field]
    G_speed = edges_and_nodes_gdf_to_graph(nodes_gdf, edges_gdf)
    
    return(G_speed)
