import os, sys, time, importlib, logging

import geopandas as gpd
import pandas as pd
import networkx as nx
import numpy as np
import shapely.wkb as wkblib

from shapely.geometry import Point, LineString
from . import core

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

def attach_traffic_data_ways(G, mb_traffic, geom_tag="Wkt", mb_from_column="FROM_NODE", mb_to_column="TO_NODE",
                                mb_speed_cols = ["min_speed","max_speed","mean_speed"]):
    """ Attach traffic data to the ways from the mapbox data, defaulted to attach 
        information from the function simplify_mapbox_csv
        
    :param G:
        networkx object to which traffic data is attached
    
    
    """
    
    cur_highways = core.edge_gdf_from_graph(G, geometry_tag = geom_tag)  
    # Create columns in highways edges dataset
    for col in mb_speed_cols:
        cur_highways[col] = 0
    # iterate through the highways dataset
    for idx, row in cur_highways.iterrows():
        nodes = row['osm_nodes']
        all_speeds = {}
        # iterate through the osm_node pairs for the current edge
        for nIdx in range(0, len(nodes) - 1):
            st_node = nodes[nIdx]
            end_node = nodes[nIdx + 1]
            try:
                # Extract the speed for the traffic for the current OSM pairs
                cur_speed = mb_traffic.loc[(mb_traffic[mb_from_column] == st_node) & 
                                           (mb_traffic[mb_to_column] == end_node)].iloc[0]
                for col in mb_speed_cols:
                    try:
                        all_speeds[col].append(cur_speed[col])
                    except:
                        all_speeds[col] = [cur_speed[col]]
            except:
                pass           
        # For the current edge, re-attach the speeds to the edges dataset
        for col in all_speeds.keys():
            cur_highways.loc[idx,col] = np.mean(all_speeds[col])
    return(cur_highways)

    
def attach_traffic_data_dense(G, traffic_data, 
                        left_on_fields = ['stnode','endnode'],
                        right_on_fields = ['FROM_NODE','TO_NODE'],
                        length_field = "length",
                        speed_field = "min_speed"):
    """ Attach the traffic information from the traffic csv to the G networkx object. 
        USED FOR THE DENSE NETWORK WHEN EDGES ARE DEFINED EXPLICITLY THE SAME AS THE MAPBOX DATA
    
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
