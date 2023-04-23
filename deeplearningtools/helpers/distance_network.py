import numpy as np
from typing import Tuple, List
from sklearn.metrics.pairwise import cosine_distances
from deeplearningtools.helpers.alignment import align_network

def percentage_different_signs_gradients(
        first_network: Tuple[np.ndarray],
        second_network: Tuple[np.ndarray],
        use_align=False,
        cost_distance=None) -> np.ndarray:
    """
    Compute the amount or percentage of weights or grandients that go in the same direction.
    The higher the percentage, the greater the distance between two layers.

    @param first_network network or list of layers
    @param second_network network or list of layers
    @param cost_distance str or function for pairwise computation of cost matrix
    @return a np.ndarray of pairwise percentage layer

    @refered from paper CMFL: Mitigating Communication Overhead for Federated Learning

    @author Youssouph FAYE@LISTIC, France
    """

    if use_align:
        second_network, alignment_costs = align_network(first_network, second_network, distance=cost_distance)

    percentages = np.array(
        [(layer_a * layer_b < 0).mean()
         for layer_a, layer_b in zip(first_network, second_network)]
    )

    return percentages

def deep_relative_trust(
        first_network: Tuple[np.ndarray],
        second_network: Tuple[np.ndarray],
        use_align=False,
        cost_distance=None,
        return_drt_product=False) -> np.ndarray:
    """


    @param first_network network or list of layers
    @param second_network network or list of layers
    @param cost_distance str or function for pairwise computation of cost matrix
    @refered from paper Deep Relative Trust
    @author Youssouph FAYE@LISTIC, France
    """

    if use_align:
        second_network, alignment_costs = align_network(first_network, second_network, distance=cost_distance)

    distances = np.array(
        [1 + (np.linalg.norm((layer_b - layer_a)) / (np.linalg.norm(layer_a)+1e-8))
         for layer_a, layer_b in zip(first_network, second_network)]
    )
    if return_drt_product:
        return (distances.prod() - 1, distances)
    else:
        distances

def deep_relative_trust_similarity(network_weights):
    nb_models=len(network_weights)
    similarity_matrix=np.zeros(shape=(nb_models, nb_models), dtype=float)

    for i in range(nb_models):
        for j in range(nb_models):
            sim=deep_relative_trust(first_network=network_weights[i],
                                    second_network=network_weights[j],
                                    return_drt_product=True)
            similarity_matrix[i,j]=sim[0]
    return similarity_matrix
        
def euclidean_norm(
        first_network: Tuple[np.ndarray],
        second_network: Tuple[np.ndarray],
        use_align=False,
        cost_distance=None) -> np.ndarray:
    """

    @param first_network network or list of layers.
    @param second_network network or list of layers
    @param cost_distance str or function for pairwise computation of cost matrix.
    @param use_align whether we align the networks before, by default false.
    @author Youssouph FAYE@LISTIC, France
    """

    if use_align:
        second_network, alignment_costs, unit_changes_per_layer = align_network(first_network, second_network, distance=cost_distance)

    euclid_norm_layers = np.array([
        np.sqrt(np.sum(np.power((layer_b - layer_a), 2)))
        for layer_a, layer_b in zip(first_network, second_network)
    ])

    return euclid_norm_layers if not use_align else (euclid_norm_layers, alignment_costs, unit_changes_per_layer)

def cosine_similarity(
        first_network: Tuple[np.ndarray],
        second_network: Tuple[np.ndarray],
        use_align=False,
        cost_distance=None) -> np.ndarray:
    """
    Compute the pairwise cosine similarity layer. If cosine is close to 1 then we can assume
    that two layers are similars.

    @param first_network network or list of layers
    @param second_network network or list of layers
    @param cost_distance str or function for pairwise computation of cost matrix
    @refered from Flexible Clustered Federated Learning

    @return a np.ndarray which contains the pairwise cosine similarity layer

    @author Youssouph FAYE@LISTIC, France
    """

    if use_align:
        second_network, alignment_costs, unit_changes_per_layer = align_network(first_network, second_network, distance=cost_distance)

    cosine_similarity_matrix = np.array([
        layer_a.flatten().dot(layer_b.flatten().T) /
        (np.linalg.norm(layer_a) * np.linalg.norm(layer_b))
        for layer_a, layer_b in zip(first_network, second_network)
    ])

    return cosine_similarity_matrix if not use_align else (cosine_similarity_matrix, alignment_costs, unit_changes_per_layer)



def flatten(weights):
    return np.hstack([w.flatten() for w in weights])

def cosine_distance(w_a: List[np.ndarray], w_b: List[np.ndarray]):
    return cosine_distances(np.array([flatten(w) for w in [w_a, w_b]]))[0, 1]

def l2(w_a: List[np.ndarray], w_b: List[np.ndarray]):
    return np.sqrt(np.sum(np.power(flatten(w_a) - flatten(w_b), 2)))

def l1(w_a: List[np.ndarray], w_b: List[np.ndarray]):
    return np.sum(abs(flatten(w_a) - flatten(w_b)))
