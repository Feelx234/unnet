

def project_with_node_view(arr, g):
    """ correctly selects the parts of the array, that are visible when g is a node view

    When using node views, centralities are still returned for the entire graph
     this function properly returns only that part of the array that is currently selected by that node view

    """
    the_filter, reverse = g.get_vertex_filter()
    if the_filter is None:
        return arr
    the_filter = the_filter.get_array().astype(bool)
    if reverse:
        the_filter = ~the_filter

    return arr[the_filter]

def _get_filter_min_(graph):
    """ Returns a boolean array with True for the minority"""
    return project_with_node_view(graph.vp.minority.get_array().astype(bool), graph)



# Taken from networkx
def attribute_ac(M):
    """Compute assortativity for attribute matrix M.

    Parameters
    ----------
    M : numpy.ndarray
        2D ndarray representing the attribute mixing matrix.

    Notes
    -----
    This computes Eq. (2) in Ref. [1]_ , (trace(e)-sum(e^2))/(1-sum(e^2)),
    where e is the joint probability distribution (mixing matrix)
    of the specified attribute.

    References
    ----------
    .. [1] M. E. J. Newman, Mixing patterns in networks,
       Physical Review E, 67 026126, 2003
    """
    try:
        import numpy
    except ImportError as e:
        raise ImportError(
            "attribute_assortativity requires " "NumPy: http://scipy.org/"
        ) from e
    if M.sum() != 1.0:
        M = M / M.sum()
    s = (M @ M).sum()
    t = M.trace()
    r = (t - s) / (1 - s)
    return r

import numpy as np
def assortativity_inner(arr):
    tmp=np.zeros(4)
    tmp[:len(arr)]=arr[:]


    arr=tmp.reshape((2,2))
    return attribute_ac(arr)
    #arr[0,0]=2 * arr[0,0]
    #arr[1,1]=2 * arr[1,1]
    #arr/=np.sum(arr)
    #print(arr)
    #e_mm=float(arr[1,1]+arr[0,1]+arr[1,0])
    #e_MM=float(arr[0,0]+arr[0,1]+arr[1,0])
    #a=np.sum(arr, axis=0)[:]
    #b=np.sum(arr, axis=1)[:]
    #E=np.sum(arr[:])
    #return  (arr[0,0]  + arr[1,1] - np.sum(a*b))/(1 - np.sum(a*b))

def assortativity(df):

    return df.edge_raw.apply(assortativity_inner)