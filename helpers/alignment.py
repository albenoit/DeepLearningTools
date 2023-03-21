import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import stats
from typing import Tuple 

######
###
# Distance Algorithms
###
######
def pearson_cost(x: np.ndarray, y: np.ndarray, sort=True) -> int:
    if sort:
        r, p_value = stats.pearsonr(np.sort(x.flatten()), np.sort(y.flatten()))
    else:
        r, p_value = stats.pearsonr(x.flatten(), y.flatten())

    return r if r is not np.nan else 0


def spearman_cost(x: np.ndarray, y: np.ndarray, sort=True) -> int:
    if sort:
        r, p_value = stats.spearmanr(
            np.sort(x.flatten()), np.sort(y.flatten()))
    else:
        r, p_value = stats.spearmanr(x.flatten(), y.flatten())

    return r if r is not np.nan else 0

def pearson(x, y):
    return 1 - pearson_cost(x, y, sort=False)

def pearson_sorted(x, y):
    return 1 - pearson_cost(x, y, sort=True)

def spearman(x, y):
    return 1 - spearman_cost(x, y, sort=False)

def spearman_sorted(x, y):
    return 1 - spearman_cost(x, y, sort=True)

def gradient(x, y):
    return np.count_nonzero(x * y < 0)

def l1(x, y):
    return np.sum(np.abs(x - y))

def l2(x, y):
    return np.sqrt(np.sum(np.power(x - y, 2)))


DISTANCE_OPTIONS = {
    'pearson': pearson,
    'pearson_sorted': pearson_sorted,
    'spearman': spearman,
    'spearman_sorted': spearman_sorted,
    'gradient': gradient,
    'l1': l1,
    'l2': l2,
}

######
###
# Alignment Algorithms
###
######


def stack(units: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Use for adding the bias to a corresponding bias
    @param units: ndarray of units of shape = (num_units, ...)
    @param bias: ndarray of bias of shape = (num_units, )
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


def _linear_sum_assignment(
    target_layer: np.ndarray,
    align_layer: np.ndarray,
    distance=None,
    ) -> Tuple[np.ndarray, int]:
    """
    The linear sum assignment problem is also known as minimum weight matching in bipartite graphs.
    We are gonna create a matrix of cost by using distance cost computation.
    
    @param target_layer layer used as a target.
    @param align_layer layer which will be aligned to the target layer.
    @param distance used for computing pairwise cost similarity of units.
    """
    
    row_size, col_size = target_layer.shape[0], align_layer.shape[0]

    if distance is None:
        distance = DISTANCE_OPTIONS['l2']
    elif type(distance) is str:
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


def align_network(
    X,
    Y,
    distance=None, extend=False, verbose=True, inplace=True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Method used for aligning a network to an other network that we can call target_network.

    @params X and Y are both networks (tuple or np.ndarray) of shape (n_layers, n_units, ...)
    @params distance is distance use to compute the cost between two units
    @params extend if true will consider the negative layer in Y
    @author Youssouph FAYE
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
