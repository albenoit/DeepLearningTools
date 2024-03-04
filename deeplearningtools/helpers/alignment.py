# ========================================
# FileName: alignment.py
# Date: 29 june 2023 - 08:00
# Author: Alexandre Benoit
# Email: alexandre.benoit@univ-smb.fr
# GitHub: https://github.com/albenoit/DeepLearningTools
# Brief: Distance Algorithms
# for DeepLearningTools.
# =========================================

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import stats
from typing import Tuple 

# ------------------------------------------------------------
# Distance Algorithms
# ------------------------------------------------------------
def pearson_cost(x: np.ndarray, y: np.ndarray, sort=True) -> float:
    r"""
    Calculate the Pearson correlation coefficient between two arrays, the measure of dissimilarity between two variables [-1:1].

    It quantifies the strength and direction of the linear association between the variables.
    The Pearson correlation coefficient can be calculated as:

    .. math::
        r = \\frac{{\sum((x_i - \overline{x})(y_i - \overline{y}))}}{{\sqrt{{\sum(x_i - \overline{x})^2}} \cdot \sqrt{{\sum(y_i - \overline{y})^2}}}}

    where:

        - :math:`\\overline{x}` and :math:`\\overline{y}` are the means of arrays x and y respectively,
        - :math:`x_i` and :math:`y_i` are the corresponding elements of arrays x and y,

    :param x: The first array.
    :type x: np.ndarray

    :param y: The second array.
    :type y: np.ndarray

    :param sort: Whether to sort the arrays before calculating the correlation coefficient.
    :type sort: bool, optional (default=True)

    :return: The Pearson correlation coefficient.
    :rtype: float
    """
    if sort:
        r, p_value = stats.pearsonr(np.sort(x.flatten()), np.sort(y.flatten()))
    else:
        r, p_value = stats.pearsonr(x.flatten(), y.flatten())

    return r if r is not np.nan else 0

def spearman_cost(x: np.ndarray, y: np.ndarray, sort=True) -> float:
    r"""
    Calculate the Spearman correlation coefficient between two arrays.

    The Spearman correlation coefficient measures the strength and direction of the monotonic relationship between two variables. 
    It is a non-parametric measure that assesses the similarity of the ranks of the corresponding elements in the arrays.
    The Spearman correlation coefficient can be calculated as:

    .. math::
        r = 1 - \\frac{6 \\sum d_i^2}{n(n^2 - 1)}

    where:

        - :math:`r` is the Spearman correlation coefficient,
        - :math:`d_i` is the difference in the ranks of corresponding elements,
        - :math:`n` is the number of elements in the arrays.

    :param x: The first array.
    :type x: np.ndarray

    :param y: The second array.
    :type y: np.ndarray

    :param sort: Whether to sort the arrays before calculating the correlation coefficient.
    :type sort: bool, optional (default=True)

    :return: The Spearman correlation coefficient.
    :rtype: float
    """
    if sort:
        r, p_value = stats.spearmanr(
            np.sort(x.flatten()), np.sort(y.flatten()))
    else:
        r, p_value = stats.spearmanr(x.flatten(), y.flatten())

    return r if r is not np.nan else 0

def pearson(x: np.ndarray, y: np.ndarray, sort=True) -> float:
    r"""
    Calculate the normalized Pearson correlation between two arrays, the measure of similarity between two variables [0:1].

    :param x: The first array.
    :type x: np.ndarray

    :param y: The second array.
    :type y: np.ndarray

    :return: The normalized Pearson correlation.
    :rtype: float
    """
    return 1 - pearson_cost(x, y, sort=False)

def pearson_sorted(x: np.ndarray, y: np.ndarray) -> float:
    r"""
    Calculate the normalized Pearson correlation between two sorted arrays.
    
    :param x: The first sorted array.
    :type x: np.ndarray

    :param y: The second sorted array.
    :type y: np.ndarray

    :return: The normalized Pearson correlation for sorted arrays.
    :rtype: float
    """
    return 1 - pearson_cost(x, y, sort=True)

def spearman(x: np.ndarray, y: np.ndarray) -> float:
    r"""
    Calculate the normalized Spearman correlation between two arrays.

    :param x: The first array.
    :type x: np.ndarray

    :param y: The second array.
    :type y: np.ndarray

    :return: The normalized Spearman correlation.
    :rtype: float
    """
    return 1 - spearman_cost(x, y, sort=False)

def spearman_sorted(x: np.ndarray, y: np.ndarray) -> float:
    r"""
    Calculate the normalized Spearman correlation between two sorted arrays.

    :param x: The first sorted array.
    :type x: np.ndarray

    :param y: The second sorted array.
    :type y: np.ndarray

    :return: The normalized Spearman correlation for sorted arrays.
    :rtype: float
    """
    return 1 - spearman_cost(x, y, sort=True)

def gradient(x: np.ndarray, y: np.ndarray) -> int:
    r"""
    Calculate the gradient dissimilarity between two arrays.

    The gradient dissimilarity between two arrays, x and y, is a measure of their dissimilarity based on the signs of their corresponding elements.
    The gradient dissimilarity can be calculated as the number of elements in the arrays where the product of the corresponding elements is negative:

    .. math::
        gradient = \sum_i [x_i \cdot y_i < 0]

    where:

        - :math:`x_i` and :math:`y_i` are the corresponding elements of arrays x and y,

    :param x: The first array.
    :type x: np.ndarray

    :param y: The second array.
    :type y: np.ndarray

    :return: The gradient dissimilarity.
    :rtype: int
    """
    return np.count_nonzero(x * y < 0)

def l1(x: np.ndarray, y: np.ndarray) -> float:
    r"""
    Calculate the L1 distance between two arrays or vectors.

    The L1 distance, also known as the Manhattan distance or taxicab distance, between two arrays (or vectors) is a measure of the absolute difference between 
    the corresponding elements of the arrays.
    The L1 distance between two arrays, x and y, of equal size, can be calculated as follows:

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
    return np.sum(np.abs(x - y))

def l2(x: np.ndarray, y: np.ndarray) -> float:
    r"""
    Calculate the L2 distance between two arrays or vectors. 
    
    The L2 distance, also known as the Euclidean distance, between two arrays (or vectors) is a measure of the distance between these two points in Euclidean space. 
    The distance L2 between two arrays, x and y, of equal size, can be calculated as follows:

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
    return np.sqrt(np.sum(np.power(x - y, 2)))

# ------------------------------------------------------------
# Distance options
# ------------------------------------------------------------

DISTANCE_OPTIONS = {
    'pearson': pearson,
    'pearson_sorted': pearson_sorted,
    'spearman': spearman,
    'spearman_sorted': spearman_sorted,
    'gradient': gradient,
    'l1': l1,
    'l2': l2,
}

# ------------------------------------------------------------
# Alignment Algorithms
# ------------------------------------------------------------

def stack(units: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Stack the units array with the bias array.

    :param units: Array of units with shape (num_units, ...)
    :type units: np.ndarray

    :param bias: Array of bias with shape (num_units, ...)
    :type bias: np.ndarray

    :return: Stacked array with shape (num_units, channels+1, weight, height)
    :rtype: np.ndarray
    """
    # add one dim in the number of channel
    # (units, channels, weight, height) => (units, channels+1, weight, height)
    shape = list(units.shape)
    shape[1] +=1 # add one additional dim
    # create new empty units_bias array
    units_bias = np.empty(shape)
    # fill units_bias with units
    units_bias[:, :-1] = units
    # then add a filter of same value as bias
    for i, b in enumerate(bias):
        shape = units.shape[2:]
        units_bias[i, -1] = np.full(shape, b)

    return units_bias

def _linear_sum_assignment(target_layer: np.ndarray, align_layer: np.ndarray, distance=None) -> Tuple[np.ndarray, int]:
    """
    Solve the linear sum assignment problem using the Hungarian algorithm.
    The problem is also known as the minimum weight matching in bipartite graphs.
    This function creates a cost matrix by computing pairwise cost similarities between units.

    :param target_layer: Layer used as the target.
    :type target_layer: np.ndarray

    :param align_layer: Layer to be aligned to the target layer.
    :type align_layer: np.ndarray

    :param distance: Distance metric used for computing pairwise cost similarity of units.
                     If None, the L2 distance will be used.
                     If a string, the corresponding pre-defined distance metric will be used.
                     If a custom callable, it should accept two np.ndarray arguments and return a float.
    :type distance: str or callable, optional

    :return: The column indices representing the optimal alignment and the total cost of the alignment.
    :rtype: Tuple[np.ndarray, int]
    """
    row_size, col_size = target_layer.shape[0], align_layer.shape[0]

    if distance is None:
        distance = DISTANCE_OPTIONS['l2']
    elif isinstance(distance, str):
        distance = DISTANCE_OPTIONS[distance]
    else:
        pass

    cost = np.zeros(shape=(row_size, col_size))

    for i in range(row_size):
        for j in range(col_size):
            # cost matrix
            cost[i][j] = distance(target_layer[i], align_layer[j])

    row_ind, col_ind = linear_sum_assignment(cost)

    return col_ind, cost[row_ind, col_ind].sum()


def align_network(X, Y, distance=None, extend=False, verbose=True, inplace=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aligns a network to another network referred to as the target network.

    :param X: The source network (tuple or np.ndarray) of shape (n_layers, n_units, ...).
    :param Y: The network to be aligned (tuple or np.ndarray) of shape (n_layers, n_units, ...).
    :param distance: The distance metric used to compute the cost between two units.
                     If None, the L2 distance will be used.
                     If a string, the corresponding pre-defined distance metric will be used.
                     If a custom callable, it should accept two np.ndarray arguments and return a float.
    :param extend: If True, consider the negative layer in Y for alignment.
    :param verbose: If True, print unit changes size during alignment.
    :param inplace: If True, align Y directly in place. If False, create a copy of Y for alignment.

    :return: The aligned network, alignment costs, and size of aligned layers.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]

    Author: Youssouph Faye.
    """
    L = len(X) # number of layers
    if not inplace:
        _align_network = [np.copy(y) for y in Y] # will contain the aligned network
    else:
        _align_network = Y # will align directly the real network 

    alignment_costs = np.zeros(shape=(L,))
    unit_changes_per_layer = np.zeros(shape=(L,))
    for l in range(L):
        if X[l].ndim > 1: # if it is not a bias layer
            if X[l].shape[0] > 1:  # there's more than one unit otherwise just avoid alignment of that unit
                # we must transpose each layer because with tensorflow the axis of units is at the end
                # (weight, height, channels, units).T => (units, channels, height, weight)
                target_layer = np.copy(X[l].T)
                align_layer = np.copy(_align_network[l].T)

                # if the next layer is a layer of bias
                if l < L - 1 and X[l+1].ndim == 1:
                    # for each unit we add its corresponding bias
                    target_layer = stack(target_layer, X[l+1])
                    align_layer = stack(align_layer, _align_network[l+1])

                # considering the negative units
                if extend:
                    # align_layer = np.concatenate((align_layer, -align_layer), axis=0)
                    pass
                # index of unit in R that match in S
                K, cost = _linear_sum_assignment(target_layer, align_layer, distance)
                # compute the number of units that have been changed or realigned
                unit_changes_size = (np.arange(len(K)) != K).sum()
                if verbose:
                    print('Unit changes size =', unit_changes_size)

                # swap i in Y such that i now matches the node in X that it matched in K
                _align_network[l].T[:] = _align_network[l].T[K]
                # update the cost of the alignment
                alignment_costs[l] = cost
                # update aligned layers size
                unit_changes_per_layer[l] = unit_changes_size
            else: # not need to align something
                K = None

        # it's probably a bias layer len(K) == X[l].size or num dim is equals to 1
        elif K is not None and X[l].ndim == 1:
            _align_network[l].T[:] = _align_network[l].T[K]
            # update the cost of the alignment
            alignment_costs[l] = cost
            # update aligned layers size
            unit_changes_per_layer[l] = unit_changes_size
            if verbose:
                print('Bias unit changes size =', unit_changes_size)

        else: # we avoid to align this layer
            pass

        # if we are not at the last layer then we must update the order of weights for each
        # unit in the next layer. The next layer must not be a biais (ndim > 1)
        # NB: if the lenght of aligned units from the current layer doesn't match with the
        # number of weights in the next layer, probably due to Pooling layer, then discard.
        # if K is None we then not need the change the order in the next layer.
        if l < L - 1 \
            and K is not None \
                and _align_network[l+1].ndim > 1 \
                    and _align_network[l+1].T.shape[1] == len(K):
            # for each unit in the next layer
            # (weight, height, channels, units).T => (units, channels, height, weight)
            # then we loop through units and then change channels orders
            for unit in _align_network[l+1].T:
                # change order of weights according to the current alignment
                unit[:] = unit[K]

        else: # keep order of weights for the next layer unchanged
            pass

    # return new aligned network, alignment costs and size of aligned layers
    return (_align_network, alignment_costs, unit_changes_per_layer)