import numpy as np

def default_distances_to_similarities(distances:np.ndarray) -> np.ndarray:
    assert len(distances.shape) == 2

    #return 1/np.log(1+distances)
    #return np.power(np.log(1+(1/(distances+1))), 2) * 100
    #return np.nanmax(distances) - distances
    return dist_to_sim__normalized_power(distances, 3)

def dist_to_sim__normalized_power(distances:np.ndarray, p:float):
    mini, maxi = np.nanmin(distances), np.nanmax(distances)
    if mini == maxi: return distances

    n_distances = 1 - (distances - mini)/(maxi - mini)
    n_distances = np.power(n_distances, p)

    return n_distances

def get_clientinfo(node):
    client_info = {'model': node['model']}
    if 'gradient' in node: client_info['gradient'] = node['gradient']
    return client_info