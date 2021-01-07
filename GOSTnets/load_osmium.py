import osmium, logging, pyproj

import shapely.wkb as wkblib
import networkx as nx
import pandas as pd

from shapely.geometry import LineString, Point
from shapely.ops import transform
from functools import partial

wkbfab = osmium.geom.WKBFactory()

# extract highways
class HighwayExtractor(osmium.SimpleHandler):
    """ Extractor for use in osmium SimpleHandler to extract nodes and highways
    """
    def __init__(self, verbose=False):
        osmium.SimpleHandler.__init__(self)
        self.verbose = verbose
        self.nodes = []
        self.raw_h = []
        self.highways = []
        self.broken_highways = []
        
    def node(self, n):
        wkb = wkbfab.create_point(n)
        shp = wkblib.loads(wkb, hex=True)
        self.nodes.append([n.id, shp, shp.x, shp.y])
    
    def way(self, w):
        self.raw_h.append(w)
        try:
            nodes = [x.ref for x in w.nodes]
            wkb = wkbfab.create_linestring(w)
            shp = wkblib.loads(wkb, hex=True)
            info = [w.id, nodes, shp, w.tags['highway']]
            self.highways.append(info)
        except:
            nodes = [x for x in w.nodes if x.location.valid()]
            if len(nodes) > 1:
                shp = LineString([Point(x.location.x, x.location.y) for x in nodes])
                info = [w.id, nodes, shp, w.tags['highway']]
                self.highways.append(info)
            else:
                self.broken_highways.append(w)
            if self.verbose:
                logging.warning("Error Processing OSM Way %s" % w.id)

def create_G(osm_file, verbose=False, project="", densify=True):
    """ Generate a networkX digraph from an osm.pbf
    
    :param osm_file: 
      path to a .osm.pbf
    :param verbose:
      [optional] boolean on weather to log information ancd errors
    :param project:
      [optional] [default ""] int to define the epsg number to project the data to in order to 
      calculate edge length in metres
    :param densify:
      [optional] if True, create new edges for every for every node combination for each way
                 if false, create edges only from the whole ways
    :returns: 
      a multidigraph object
    """
    h = HighwayExtractor(verbose)
    h.apply_file(osm_file, locations=True)
    
    all_h = []
    #return(h)
    nodes_df = pd.DataFrame(h.nodes, columns = ["osm_id", "geometry", "x", "y"])   
    if project != '':
        project_WGS = partial(
            pyproj.transform,
            pyproj.Proj(init='epsg:4326'),
            pyproj.Proj(init=f'epsg:{project}'))
    if densify:
        for x in h.highways:
            for n_idx in range(0, (len(x[1]) - 1)):
                try:
                    osm_id_from = x[1][n_idx].ref
                except:
                    osm_id_from = x[1][n_idx]
                try:
                    osm_id_to   = x[1][n_idx+1].ref
                except:
                    osm_id_to   = x[1][n_idx+1]
                try:
                    from_pt = nodes_df.loc[nodes_df['osm_id'] == osm_id_from,'geometry'].iloc[0]
                    to_pt   = nodes_df.loc[nodes_df['osm_id'] == osm_id_to  ,'geometry'].iloc[0]
                    edge = LineString([from_pt, to_pt])
                    if project != '':
                        edge_proj = transform(project_WGS, edge)
                    else:
                        edge_proj = edge
                    attr = {'osm_id':x[0], 'Wkt':edge, 'length':edge_proj.length, 'infra_type':x[3]}
                    #Create an edge from the list of nodes in both directions
                    all_h.append([osm_id_from, osm_id_to, attr])
                    all_h.append([osm_id_to, osm_id_from, attr])
                except:
                    if verbose:
                        logging.warning(f"Error adding edge between nodes {osm_id_from} and {osm_id_to}")
        #Create and populate the networkx graph
        G = nx.MultiDiGraph()        
        used_nodes = [[osm_id, {'shape':shp, 'x':x, 'y':y}] for osm_id, shp, x, y in h.nodes]        
        G.add_nodes_from(used_nodes)
        G.add_edges_from(all_h)
    else:
        all_nodes = []
        for x in h.highways:
            edge = x[2]
            if project != '':
                edge_proj = transform(project_WGS, edge)
            else:
                edge_proj = edge
            osm_node_from = x[1][0]
            osm_node_to = x[1][-1]
            all_nodes.append(osm_node_from)
            all_nodes.append(osm_node_to)
            attr = {'osm_id':x[0], 'Wkt':edge, 'length':edge_proj.length, 'infra_type':x[3], 'osm_nodes':x[1]}
            all_h.append([osm_node_from, osm_node_to, attr])
            all_h.append([osm_node_to, osm_node_from, attr])
        # create nodes dataset from 
        all_nodes = list(set(all_nodes))
        #used_nodes = [[osm_id, {'shape':shp, 'x':x, 'y':y}] for osm_id, shp, x, y in h.nodes]
        
        nodes_df = nodes_df[nodes_df['osm_id'].isin(all_nodes)]
        used_nodes = [[row['osm_id'], {'shape':row['geometry'], 'x':row['x'], 'y':row['y']}] for idx, row in nodes_df.iterrows()]        
        G = nx.MultiDiGraph()              
        G.add_nodes_from(used_nodes)
        G.add_edges_from(all_h)
    return(G)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    