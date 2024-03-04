# ========================================
# FileName: distance_network.py
# Date: 29 june 2023 - 08:00
# Author: Alexandre Benoit
# Email: alexandre.benoit@univ-smb.fr
# GitHub: https://github.com/albenoit/DeepLearningTools
# Brief: A collection of distance metrics
# for DeepLearningTools.
# =========================================

import numpy as np
from typing import Tuple, List
from sklearn.metrics.pairwise import cosine_distances
from deeplearningtools.helpers.alignment import align_network

def percentage_different_signs_gradients(
        first_network: Tuple[np.ndarray],
        second_network: Tuple[np.ndarray],
        use_align=False,
        cost_distance=None) -> np.ndarray:
    r"""
    Compute the amount or percentage of weights or gradients that go in the same direction.
    The higher the percentage, the greater the distance between two layers.
    Referenced from the paper "CMFL: Mitigating Communication Overhead for Federated Learning" (https://home.cse.ust.hk/~weiwa/papers/cmfl-icdcs19.pdf).

    The percentage of weights or gradients that go in the same direction can be computed using the following equation:

    .. math::
        e(u,\overline{u}) = \\frac{{1}}{{N}} \sum_{j=1}^{N} I(sign(u_{j}) = sign(\overline{u}_{j}))

    where:

        - :math:`u=<u_1, u_2, ..., u_N>` is the local update,
        - :math:`N` is number of model parameters,
        - :math:`I\\text{{sign}}(u_{j} = \overline{u}_{j})=1` if :math:`u_j` and :math:`\overline{u}_{j}` are the same sign, and :math:`0` otherwise.

    :param first_network: Network or list of layers.
    :param second_network: Network or list of layers.
    :param use_align: Boolean indicating whether to align the networks.
    :param cost_distance: String or function for pairwise computation of the cost matrix.
    :return: Array of pairwise percentage layer.

    Author: Youssouph Faye.
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
    r"""
    Compute the deep relative trust between two networks or list of layers.

    Referenced from the paper "Deep Relative Trust" (https://arxiv.org/abs/2002.03432).

    This criterion computes the absolute difference between the projected trust values for each weight pair from the two networks, and normalizes this sum by the product of the number of layers and the dimension of the layers.
    The deep relative trust between two networks or list of layers can be computed using the following equation:

    .. math::
        \\left| \\frac{{f(x) - \\tilde{f}(x)}}{{f(x)}} \\right| \leq \left(1 + \\frac{{|\Delta a|}}{{|a|}}\\right) \left(1 + \\frac{{|\Delta b|}}{{|b|}}\\right) - 1

    where:

        - :math:`f(x)` is the output of the first network or list of layers.
        - :math:`\\tilde{f}(x)` is the output of the second network or list of layers.
        - :math:`\Delta a` represents the perturbation in the parameter `a`.
        - :math:`\Delta b` represents the perturbation in the parameter `b`.
        - :math:`a` and :math:`b` are the original parameters.

    :param first_network: Network or list of layers.
    :param second_network: Network or list of layers.
    :param use_align: Boolean indicating whether to align the networks.
    :param cost_distance: String or function for pairwise computation of the cost matrix.
    :param return_drt_product: Boolean indicating whether to return the deep relative trust product.
    :return: Array of distances or deep relative trust product and distances.

    Author: Youssouph Faye.
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
        return distances

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
    r"""
    Compute the Euclidean norm between two networks or list of layers.

    The Euclidean norm can be calculated using the following equation:

    .. math::
        ||w_b - w_a|| = \sqrt{\sum{(w_b - w_a)^2}}

    :param first_network: Network or list of layers.
    :param second_network: Network or list of layers.
    :param use_align: Boolean indicating whether to align the networks.
    :param cost_distance: String or function for pairwise computation of the cost matrix.
    :return: Array of Euclidean norm for each layer.

    Author: Youssouph Faye.
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
    r"""
    Compute the pairwise cosine similarity between two networks or list of layers.

    Referred from the paper "Flexible Clustered Federated Learning" (https://arxiv.org/pdf/2108.09749.pdf).
    The cosine similarity is calculated as the cosine of the angle between the two weight vectors:

    .. math::
        {cosine\\_{similarity}} = \\frac{{w_a \\cdot w_b}}{{\\|w_a\\| \\cdot \\|w_b\\|}}

    where:
        
        -:math:`w_a \cdot w_b` denotes the weight vectors
    
    :param first_network: Network or list of layers.
    :param second_network: Network or list of layers.
    :param use_align: Boolean indicating whether to align the networks.
    :param cost_distance: String or function for pairwise computation of the cost matrix.
    :return: Array of pairwise cosine similarity for each layer.

    Author: Youssouph Faye.
    """
    if use_align:
        second_network, alignment_costs, unit_changes_per_layer = align_network(first_network, second_network, distance=cost_distance)

    cosine_similarity_matrix = np.array([
        layer_a.flatten().dot(layer_b.flatten().T) /
        (np.linalg.norm(layer_a) * np.linalg.norm(layer_b))
        for layer_a, layer_b in zip(first_network, second_network)
    ])

    return cosine_similarity_matrix if not use_align else (cosine_similarity_matrix, alignment_costs, unit_changes_per_layer)

def flatten(weights: List[np.ndarray]) -> np.ndarray:
    """
    Flatten a list of weights arrays into a single 1D array.

    :param weights: List of weights arrays.
    :return: Flattened array of weights.
    """
    return np.hstack([w.flatten() for w in weights])

def cosine_distance(w_a: List[np.ndarray], w_b: List[np.ndarray]) -> float:
    r"""
    Compute the cosine distance between two sets of weights.

    The cosine distance between two sets of weights, w_a and w_b, can be calculated using the following equation:

    .. math::
        {cosine\\_{distance}} = 1 - {cosine\\_{similarity}}

    :param w_a: List of weight arrays for the first set.
    :param w_b: List of weight arrays for the second set.
    :return: Cosine distance between the two sets of weights.
    """
    return cosine_distances(np.array([flatten(w) for w in [w_a, w_b]]))[0, 1]

def l2(w_a: List[np.ndarray], w_b: List[np.ndarray]) -> float:
    r"""
    Calculate the L2 distance between two arrays or vectors. 
    
    The L2 distance, also known as the Euclidean distance, between two arrays (or vectors) is a measure of the distance between these two points in Euclidean space. 
    Mathematically, the distance L2 between two arrays, x and y, of equal size, can be calculated as follows:

    .. math::
        L2_{distance} = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}

    where:

        - :math:`x_i` and :math:`y_i` are the corresponding elements of arrays x and y,

    :param x: The first array.
    :type x: np.ndarray

    :param y: The second array.
    :type y: np.ndarray

    :return: The L2 distance.
    :rtype: float
    """
    return np.sqrt(np.sum(np.power(flatten(w_a) - flatten(w_b), 2)))

def l1(w_a: List[np.ndarray], w_b: List[np.ndarray]) -> float:
    r"""
    Calculate the L1 distance between two arrays or vectors.

    The L1 distance, also known as the Manhattan distance or taxicab distance, between two arrays (or vectors) is a measure of the absolute difference between 
    the corresponding elements of the arrays.
    Mathematically, the L1 distance between two arrays, x and y, of equal size, can be calculated as follows:

    .. math::
        L1_{distance} = \sum_{i=1}^{n} |x_i - y_i|

    where:

        - :math:`x_i` and :math:`y_i` are the corresponding elements of arrays x and y,
        
    :param x: The first array.
    :type x: np.ndarray

    :param y: The second array.
    :type y: np.ndarray

    :return: The L1 distance.
    :rtype: float
    """
    return np.sum(abs(flatten(w_a) - flatten(w_b)))
