import matplotlib
matplotlib.use('Agg') # forbids display, all figures must be saved as image files
import matplotlib.pyplot as plt

from deeplearningtools.helpers.distance_metrics.metrics import ClientMatrixMetric
import matplotlib.colors as mcolors
from matplotlib.colors import ColorConverter
from netgraph import Graph
import os
import networkx as nx
import pickle
import math
import numpy as np

LOGS_PATH = "FlexCLF_logs/"
COLOR_IDS=list(mcolors.TABLEAU_COLORS.values())


class GraphCFL_Visualisation:
    def __init__(self, sim_metric:ClientMatrixMetric):
        self.node_sizes = {}
        self.node_colors={}
        self.node_alphas = {}
        self.edge_color=[]
        self.edge_width = {}
        self.edge_colors = []
        self.edge_scale = 2
        self.scale = 1.1
        self.sim_metric = sim_metric

    def clusters_to_communities(self, clusters):
        """ given 'clusters', a list of clusters that list each of their node,
            returns a dictionnary that maps each node to its community id
        """
        node_to_community = {}
        for cluster_id, cluster in enumerate(clusters):
            for node in cluster:
                node_to_community[node] = cluster_id
        return node_to_community


    def REPLACED_BY_UPDATE_CLUSTER_GRAPH_update_with_dict(self, graph, rnd: int, clusters: [], clients: dict, sizes: dict):
        """
        nodes: nodes of the graph
        sizes: dictionnary that stores several graph evaluations
        """
        nodes_id = []
        for client in clients:
            client_id = client.id
            nodes_id.append(str(client_id))
        for node_id in range(len(nodes_id)):
            print("sizes[node_id] : ", sizes[node_id])
            print("self.scale : ", self.scale)
            self.node_sizes[node_id] = max(min(sizes[node_id] * (self.scale + 3), 10), 5)
        self._update_edges_and_nodes(graph, rnd, clusters)

    def update_cluster_graph(self, subgraph, rnd, fullgraph, metric_size):
        self.node_sizes={}#create a new dict in case the graph has changed from previous round
        self.node_colors={}
        for cluster in fullgraph.nodes:
            self.node_sizes[cluster] = max(min(metric_size[cluster] * (self.scale + 3), 10), 5)
        clusters_list = [[cluster] for cluster in subgraph.nodes]
        self._update_edges_and_nodes(subgraph, rnd, clusters_list)

    def REPLACED_BY_UPDATE_CLIENT_GRAPH_update_with_func(self, graph, rnd: int, clusters: [], clients: dict, centroids: dict, size_function):
        node_to_community = self.clusters_to_communities(clusters)
        # bigger it is, closer the node is from the center of the cluster
        for client_a in clients:
            for i, centroid in enumerate(centroids):
                if i == node_to_community[client_a.id]:
                    distance = self.sim_metric.model_distance(client_a.data, centroid)
                    self.node_sizes[client_a.id] = max(min(size_function(distance) * 4, 10),5) * self.scale
        self._update_edges_and_nodes(graph, rnd, clusters)

    def update_client_graph(self, subgraph: nx.Graph, rnd: int, fullgraph:nx.MultiGraph, communities, size_function):
        """ update the visualisation of the graph
        Arguments:
            subgraph: the subgraph to plot
            rnd: the round index
            fullgraph: the full graph of the clients
            communities: the list of communities as a list of clusters that contain a list of client ids 
            size_function: the function that will be used to compute the size of the nodes
        """
        # bigger it is, closer the node is from the center of the cluster

        # @FIX process functions
        for client in fullgraph.nodes:
            #get the distance to the closest cluster
            #print('update_graph::client', client, fullgraph.nodes[client])
            #print('--> closest cluster', fullgraph.nodes[client]["closest_clusters"].iloc[0]['dist'])
            distance=fullgraph.nodes[client]["closest_clusters"].iloc[0]['dist']
            # @debug scalar to array and convert result to scalar

            scalar = size_function(np.reshape(distance, (1,1)))[0][0]
            self.node_sizes[client] = max(min(scalar * 4, 10),4) * self.scale
        #print('update_graph::Plotting communities', communities)
        self._update_edges_and_nodes(subgraph, rnd, communities)
        
    def update_colors(self, clusters: dict, colors: dict):
        for node_id in colors:
            for i, cluster in enumerate(clusters):
                if node_id in cluster:
                    self.node_colors[i] = colors[node_id]

    def get_node_colors(self):
        return self.node_colors

    def _update_edges_and_nodes(self, graph, rnd:int, clusters:dict):
        """
        extract subgraph of interest for plotting
        Arguments: 
            rnd, the round index
            clusters, the list of clusters with
        """
        #get all nodes and prepare their appearance
        #node_color = {node: community_to_color[community_id] for node, community_id in node_to_community.items()}
        print("[_update_edges_and_nodes]")
        print(graph.nodes())
        for node in graph.nodes():
            cluster_id=-1
            alpha=0.4
            node_rnd = nx.get_node_attributes(graph, "last_round")[node]
            self.node_alphas[node] = 1.0 - min((rnd - node_rnd) / 4, 0.8)
            for i, cluster in enumerate(clusters):
                for client_id in cluster:
                    if node == client_id:
                        ###print('node', node, '-> associated to cluster ', cluster_key)
                        cluster_id = i
                        alpha=1.
                        break
                if cluster_id!=-1:
                    break
            self.node_colors[node] = ColorConverter().to_rgba(COLOR_IDS[cluster_id%10], alpha=alpha)

        #normalize edge weight
        maxi = -math.inf 
        for u, v, data in graph.edges(data=True):
            if data["weight"] > maxi:
                maxi = data["weight"]
        #get all edges of interest and prepare their appearance 
        for u, v, data in graph.edges(data=True):
            self.edge_width[(u, v)] = min(data["weight"] * self.edge_scale, 10)
            # !! OLD EDGE COLORATION IS NOT WORKING BECAUSE OF THE NEW VISUALISATION PACKAGE !!
            if 'last_round' in data.keys():
                if data['last_round']==rnd:
                    self.edge_colors.append('orange')
                else: #print old links in blue with decreasing alfa as they get older
                    self.edge_colors.append(ColorConverter().to_rgba('blue', alpha=(1-(rnd-data['last_round'])/rnd)))
            else:
                self.edge_color.append(ColorConverter().to_rgba('green', alpha=0.5))
        self.edge_colors.append(ColorConverter().to_rgba('green', alpha=0.5))


    def save_gml_graph(self, subgraph, clusters, rnd, name, subplt: int)-> None:
        """ helper that plots the instance client graph based on a given edge type
        maybe look here for plotting solutions: https://www.malinga.me/networkx-visualization-with-graphviz-example/
        arguments:
        """

        ############# plot the graph
        #plt.figure(figsize=(12.80, 10.24))
        plt.subplot(subplt)
        #plt.tight_layout()

        node_to_community = self.clusters_to_communities(clusters)
        #print("len(edge_widths) : ", len(self.edge_width))
        #print("node_to_community: ", node_to_community)
        #print("graph nodes: ", subgraph.nodes)

        if len(self.node_sizes) != len(self.node_colors):
            print("node_colors : ", self.node_colors)
            print("node_sizes : ",self.node_sizes)
            print("len(node_size) : ",len(self.node_sizes))
            print("len(node_colors) : ", len(self.node_colors))
            raise ValueError("len(self.node_sizes) != len(self.node_colors)")
        if len(subgraph.nodes) != len(node_to_community):
            raise ValueError("len(subgraph.nodes) != len(node_to_community)")
            
        print('Preparing visualization graph... this step may be long')
        # to comment -> deactivate graph plotting (long processing) @debug
        Graph(subgraph, node_labels=True,
            node_color=self.node_colors, edge_width=self.edge_width, edge_alpha=0.1, edge_color="#0c0c0d",
            node_size= self.node_sizes, node_alpha=self.node_alphas,
            node_layout='community', node_layout_kwargs=dict(node_to_community=node_to_community),
            edge_layout='curved', edge_layout_kwargs=dict(k=2000),
        )
        print('Graph prepared')
        """
        edge_layout=
        - 'straight' : draw edges as straight lines
        - 'curved'   : draw edges as curved splines; the spline control points are optimised to avoid other nodes and edges
        - 'arc'      : draw edges as arcs with a fixed curvature
        - 'bundled'  : draw edges as edge bundles (ndlr: makes clean graph but hard to see edges)
        """

        ############# saving graph to file
        # @debug -> does not work with attr models in nodes
        #nx.write_gml(subgraph, "subgraph_r"+name+"_"+str(rnd)+".gml") #reminder, reload graph: mygraph = nx.read_gml("path.to.file")
        #using pickle
        with open('subgraph_'+name+"_"+str(rnd)+'.gpickle', 'wb') as f:
            pickle.dump(subgraph, f, pickle.HIGHEST_PROTOCOL)
        #reminder related loader: with open('filename.pickle', 'rb') as f: mynewloadedgraph = pickle.load(f)
        print('Graph saved')

    def save_png_graph(self, selected_edges, rnd, name, title=None)-> None:
        """ helper that plots the instance client graph based on a given edge type
        maybe look here for plotting solutions: https://www.malinga.me/networkx-visualization-with-graphviz-example/
        arguments:
        """
        ############# save the figure on the disk
        plt.axis('off')
        if title is None:
            plt.title('Graph at round '+str(rnd) +' edges:'+selected_edges)
        else:
            plt.title(title)
        #plt.savefig(os.path.join(os.getcwd(),LOGS_PATH,'clients_graph_'+name+"_r"+str(rnd)+'.svg'))
        plt.savefig(os.path.join(os.getcwd(),LOGS_PATH,'clients_graph_'+name+"_r"+str(rnd)+'.png'), dpi=400)
        plt.clf()

    def add_text(self, text)-> None:
        """ helper that plots the instance client graph based on a given edge type
        maybe look here for plotting solutions: https://www.malinga.me/networkx-visualization-with-graphviz-example/
        arguments:
        """
        ############# add text on the figure
        plt.text(50, 50, text, fontsize=18, bbox=dict(facecolor='red', alpha=0.5))

    #######################################################
