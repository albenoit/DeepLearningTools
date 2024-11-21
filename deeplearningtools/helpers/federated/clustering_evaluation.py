from deeplearningtools.helpers.distance_metrics.metrics import ClientMatrixMetric
from deeplearningtools.tools.average_calculator import Average_calculator
from sklearn.metrics import adjusted_rand_score
from cdlib import evaluation
from statistics import mean
import networkx as nx
import numpy as np
import statistics
import json
import os

class ClusteringEvaluator:
    def __init__(self, graph, communities, sim_metric:ClientMatrixMetric):
        self.graph = graph
        self.communities = communities
        self.sim_metric = sim_metric

    def avg_embeddedness(self):
        return self.communities.avg_embeddedness(summary=False)

    def size(self):
        if self.communities is None:
            raise "Error: there is no communities to work with, please define them before calling this method"
        return evaluation.size(self.graph, self.communities, summary=False)

    def avg_distance(self):
        if self.communities is None:
            raise "Error: there is no communities to work with, please define them before calling this method"
        return evaluation.avg_distance(self.graph, self.communities, summary=False) # doesnt seem to take the similarity value as a distance. avg is always 1 :-(

    def _get_average_distance_from_node(self, cid, nid):
        neighbours_id = self.communities.communities[cid]
        average = 0
        for neighbour in neighbours_id:
            neighbour_pos = nx.get_node_attributes(self.graph, "model")[neighbour]
            node_pos = nx.get_node_attributes(self.graph, "model")[nid]
            distance = self.sim_metric.model_distance(neighbour_pos, node_pos)
            average += distance
        average /= len(neighbours_id)
        return float(average)

    def _get_centroid(self, cid):
        community = self.communities.communities[cid]
        weights = []
        for node in community:
            node_model = nx.get_node_attributes(self.graph, "model")[node]
            weights.append(node_model)
        ac = Average_calculator()
        return ac.mean(weights)

    def _get_nearest_cluster(self, cid):
        # process centroids
        centroids = []
        for count, community in enumerate(self.communities.communities):
            centroids.append(self._get_centroid(count))
        distances = {}
        # process distances between centroids
        for i in range(len(centroids)):
            if i != cid:
                distances[i] = self.sim_metric.model_distance(centroids[cid], centroids[i])
        # find the nearest centroid from cid
        if not distances: return None
        min_val = min(distances.values())
        return [k for k in distances if distances[k] == min_val][0]

    def silhouette_score(self):
        silhouette_scores = []
        nodes = self.graph.nodes
        nearest_clusters = {}
        if len(self.communities.communities) > 1:
            for i in range(len(self.communities.communities)):
                nearest_clusters[i] = self._get_nearest_cluster(i)

            for node in nodes:
                # get the cluster id of this node
                cluster_id = None
                for count, community in enumerate(self.communities.communities):
                    if node in community:
                        cluster_id = count
                        break
                # process the average distance to every nodes
                a = self._get_average_distance_from_node(cluster_id, node)
                # récupérer le cluster le plus proche
                nearest_cid = nearest_clusters[cluster_id]
                # calculer pour chaque noeud la distance moyenne à eux
                b = self._get_average_distance_from_node(nearest_cid, node)
                # process the silhouette score of "node"
                silhouette_scores.append((b - a) / (max(a, b)+1e-6))
            return float(mean(silhouette_scores))
        else:
            return 0

    def avg_intracluster_similarity(self):
        """
        Process the intracluster average distance
        """
        if self.communities is None:
            raise "Error: there is no communities to work with, please define them before calling this method"
        edges = [edge for edge in self.graph.edges]
        values = []
        # get all communities
        for community in self.communities.communities:
            avg_distance = -1
            # get edges between all nodes of this community
            for node1 in community:
                for node2 in community:
                    if node1 != node2:
                        # sum their weight (should be the trusted_distance)
                        attributes = self.graph.get_edge_data(node1, node2)
                        if not attributes is None:
                            avg_distance += attributes["weight"]
                        else:
                            avg_distance += 0
                        
            # average their weight and return the result
            if avg_distance != -1:
                values.append(avg_distance/len(community))

        #print("values intra: ", values)
        #print("statistics.mean(intra) : ", statistics.mean(values))
        return float(statistics.mean(values)) if len(values) > 0 else -1

    def avg_internode_similarity(self):
        '''
        Process the internode average distance
        If used on the cluster graph -> it processes the intercluster distances
        '''
        edges = [edge for edge in self.graph.edges]
        values = []
        # get edges between all nodes
        avg_distance = 0
        for node1 in self.graph.nodes:
            for node2 in self.graph.nodes:
                if node1 != node2:
                    # sum their weight (should be the trusted_distance)
                    attributes = self.graph.get_edge_data(node1, node2)
                    if not attributes is None:
                        avg_distance += attributes["weight"]
                    else:
                        avg_distance += 0
            # average their weight and return the result
        if len(self.graph.edges) > 0:
            values.append(avg_distance/len(self.graph.edges))
            return float(statistics.mean(values))
        #print("values inter: ", values)
        #print("statistics.mean(inter) : ", statistics.mean(values))
        
        return float(statistics.mean(values)) if len(values) > 0 else -1


    def intra_inter_similarity_ratio(self, inter_values, intra_values):
        '''
        Process the intra-inter cluster average distance
        '''
        if inter_values == -1 or intra_values == -1 : 
            print('Exception : intra or inter could not be run') 
            return -1
        print("intra_values -> ", intra_values)    
        print("inter_values -> ", inter_values)    
        return float(intra_values / inter_values)
                
    def _community_to_partitioning(self):
        # replace each client id by its cluster number
        sumlen = 0
        for c in self.communities.communities:
            sumlen += len(c)
        clustering = [0 for i in range(sumlen)]
        try:
            for cnt, community in enumerate(self.communities.communities):
                for value in community:
                    clustering[int(value)] = cnt
            return clustering
        except:
            raise Exception

    def mnist_adjusted_rand_index(self):
        # Works only for a specific distribution (mnist_100)
        ground_truth = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        try:
            clustering = self._community_to_partitioning()
            return adjusted_rand_score(ground_truth, clustering)
        except:
            print("Not enough clients to process the Adjusted Rand Index")
            return -1

    def cifar10_adjusted_rand_index(self):
        # Works only for the animals/vehicles clustering
        ground_truth = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
        try:
            clustering = self._community_to_partitioning()
            return adjusted_rand_score(ground_truth, clustering)
        except:
            print("Not enough clients to process the Adjusted Rand Index")
            return -1

    def cifar10_purity(self):
        # load cifar10 majority classes to process the purity
        target_path=os.path.join('/uds_data/listic/mickaelb/federated_cifar10_data/train/')
        filename = target_path + "/majority_label.json"
        f = open(filename)
        majority_class = json.loads(f.read())
        f.close()
        print("reading majority_class (evaluation) -> ", majority_class)
        expected = {}
        expected[0] = []
        expected[1] = []
        for key in majority_class:
            if majority_class[key] == 0 or majority_class[key] == 1 or majority_class[key] == 8 or majority_class[key] == 9:
                expected[0].append(key)
            else:
                expected[1].append(key)
        print("expected clustering -> ", expected)
        print("self.communities -> ", self.communities.communities)

        '''
        expected = {0: ['2','3', '4', '5', '6', '7'],
        1: ['0', '1', '8', '9']}
        '''

        vpurity = 0
        for community in self.communities.communities:
            vmax = 0
            for key in expected:
                # measure the similarity of the community to the expected list
                intersection_list = list(set(community) & set(expected[key]))
                v = len(intersection_list) / len(community)
                if v > vmax:
                    vmax = v
            print("community max value : ", community ," -> ", vmax)
            vpurity += vmax
        # average purity of our communities
        vpurity /= len(self.communities.communities)
        print("clustering purity -> ", vpurity)
        return float(vpurity)

    def cifar100_purity(self):
        # coarse_id: [fine_ids]
        expected = {0: ['4', '30', '55', '72', '95'],
        1: ['1', '32', '67', '73', '91'],
        2: ['54', '62', '70', '82', '92'],
        3: ['9', '10', '16', '28', '61'],
        4: ['0', '51', '53', '57', '83'],
        5: ['22', '39', '40', '86', '87'],
        6: ['5', '20', '25', '84', '94'],
        7: ['6',' 7', '14', '18', '24'],
        8: ['3', '42', '43', '88', '97'],
        9: ['12', '17', '37', '68', '76'],
        10: ['23', '33', '49', '60', '71'],
        11: ['15', '19', '21', '31', '38'],
        12: ['34', '63', '64', '66', '75'],
        13: ['26', '45', '77', '79', '99'],
        14: ['2', '11', '35', '46', '98'],
        15: ['27', '29', '44', '78', '93'],
        16: ['36', '50', '65', '74', '80'],
        17: ['47', '52', '56', '59', '96'],
        18: ['8', '13', '48', '58', '90'],
        19: ['41', '69', '81', '85', '89']}

        vpurity = 0
        for community in self.communities.communities:
            vmax = 0
            for key in expected:
                # measure the similarity of the community to the expected list
                intersection_list = list(set(community) & set(expected[key]))
                v = len(intersection_list) / len(expected[key])
                if v > vmax:
                    vmax = v
            vpurity += vmax
        # average purity of our communities
        vpurity /= len(self.communities.communities)
        return float(vpurity)



