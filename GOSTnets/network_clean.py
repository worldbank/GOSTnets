#the cleaning network part
import os, sys, time

import importlib
import osmnx as ox

from shapely.ops import unary_union
from shapely.wkt import loads
from shapely.geometry import LineString, MultiLineString, Point

from .core import *


def clean_network(G, wpath = '', output_file_name = '', UTM = 'epsg:3857', WGS = {'init': 'epsg:4326'}, junctdist = 50, verbose = False):   
    """
    Topologically simplifies an input graph object by collapsing junctions and removing interstital nodes
    :param G: a graph object containing nodes and edges. Edges should have a property called 'Wkt' containing geometry objects describing the roads.
    :param wpath: the write path - a drive directory for inputs and output
    :param output_file_name: This will be the output file name with '_processed' appended
    :param UTM: The epsg code of the projection, in metres, to apply the junctdist
    :param WGS: the current crs of the graph's geometry properties. By default, assumes WGS 84 (epsg 4326)
    :param junctdist: distance within which to collapse neighboring nodes. simplifies junctions. Set to 0.1 if not simplification desired. 50m good for national (primary / secondary) networks
    :param verbose: if True, saves down intermediate stages for dissection
    """

    # Finds and deletes interstital nodes based on node degree
    #a = custom_simplify(G)

    #print('finished with custom_simplify')

    # Squeezes clusters of nodes down to a single node if they are within the snapping tolerance
    #b = simplify_junctions(G, UTM, WGS, junctdist)

    # to-do: new lengths need to be calculated for edges that are affected by simplify_junctions
    print('finished with simplify_junctions')

    # ensures all streets are two-way
    # why isn't this done after custom_simplify? and shouldn't it just be done if the road isn't one-way?
    #a = add_missing_reflected_edges(a)

    #print('finished with add_missing_reflected_edges')
    
    # save progress
    # if verbose is True: 
    #     save(a, 'a', wpath)
    
    # Finds and deletes interstital nodes based on node degree
    c = custom_simplify(G)

    print('finished with custom_simplify')
    
    # rectify geometry
    # for u, v, data in b.edges(data = True):
    #     if type(data['geometry']) == list:
    #             data['geometry'] = unbundle_geometry(data['geometry'])

    #print('finished with rectify geometry')

    # save progress
    # if verbose is True: 
    #     save(b, 'b', wpath)
    
    # For some reason CustomSimplify doesn't return a MultiDiGraph. Fix that here
    #c = convert_to_MultiDiGraph(b)

    # This is the most controversial function - removes duplicated edges. This takes care of two-lane but separate highways, BUT
    # destroys internal loops within roads. Can be run with or without this line
    #c = remove_duplicate_edges(c)

    # Run this again after removing duplicated edges
    #c = custom_simplify(c)
    
    # rectify geometry again
    # for u, v, data in c.edges(data = True):
    #     if type(data['Wkt']) == list:
    #         data['Wkt'] = unbundle_geometry(data['Wkt'])

    # Ensure all remaining edges are duplicated (two-way streets)
    #c = add_missing_reflected_edges(c)
    
    # save final
    # if verbose:
    #    save(c, '%s_processed' % output_file_name, wpath)
    
    print('Edge reduction: %s to %s (%d percent)' % (G.number_of_edges(), 
                                               c.number_of_edges(), 
                                               ((G.number_of_edges() - c.number_of_edges())/G.number_of_edges()*100)))
    return c
