from unnet.DynamicSampler import DynamicSampler
import numpy as np
from numba import njit
from numba.typed import List

@njit
def contains(arr, value):
    for i in arr:
        if i==value:
            return True
    return False

@njit
def lin_log_homo_inner(N, m , minority_fraction, homophily, epsilon, labels, edges):
    """
    samples edges according to barabasi + homophily uses sampling without replacement to avoid multi edges
    Runtime: This algorithm runs in O(m * n log(n))
    """
    n_edges = 0
    minority_size = int(minority_fraction * N)
    majority_size = N - minority_size

    i_min=0
    i_maj=0
    n_random = 0


    min_tree = DynamicSampler(minority_size)
    maj_tree = DynamicSampler(majority_size)
    index2node_min=np.zeros(minority_size, dtype=np.uint32)
    index2node_maj=np.zeros(majority_size, dtype=np.uint32)
    node2index=np.zeros(N, dtype=np.uint32)
    i_min = 0
    i_maj = 0

    for j in range(N):
        if labels[j]:
            index2node_min[i_min]=j
            node2index[j]=i_min

            min_tree.insert(np.uint32(i_min),0)
            i_min+=1
        else:
            index2node_maj[i_maj]=j
            node2index[j]=i_maj

            maj_tree.insert(i_maj,0)
            i_maj+=1
            
    targets = np.zeros(m, dtype=np.uint32)
    source = m
    for source in range(m, N):
        #print("<<<<", source)
        ### Figure out which labels to attach to:
        if labels[source]:# belongs to minority
            f_min = homophily
            f_maj = 1-homophily
        else:
            f_min = 1-homophily
            f_maj = homophily

        removed_min = List()
        removed_maj = List()
        i=0
        while i < len(targets):
            #print(i)
            if maj_tree._n_items > 0:
                tickets_maj = maj_tree._tree[0]
            else:
                tickets_maj = 0
            if min_tree._n_items > 0:
                tickets_min = min_tree._tree[0]
            else:
                tickets_min = 0
            min_term = tickets_min * f_min
            maj_term = tickets_maj * f_maj
            
            #print(i, min_term, maj_term, epsilon_term)
            
            

            if maj_term <= 10.0**-6 and min_term <= 10.0**-6:
                # Need to assign randomly
                j=0
                while i < len(targets):
                    x = np.random.randint(0, source)
                    if not contains(targets[:i], x):
                        targets[i] = x
                        j+=1
                        i+=1
                n_random+=j
                continue
                    
                    
                    
            r=np.random.rand() * (min_term + maj_term)
            t=np.uint32(0)
            if r <= maj_term:
                if f_maj <=0:
                    assert False
                    #print("B")
                    continue
                index = maj_tree.sample()
                #index = maj_tree.find_for_sample((r-min_term)/(f_maj))

                val = maj_tree.remove(index)
                removed_maj.append((index, val))
                t = index2node_maj[index]
            else:
                if f_min <=0:
                    assert False
                    #print("C")
                    continue
                index = min_tree.sample()
                #index = min_tree.find_for_sample(r/(f_min))

                val = min_tree.remove(index)
                removed_min.append((index, val))
                t = index2node_min[index]

            if not contains(targets[:i], t):
                targets[i] = t
                i+=1
            #else:
            #    print(t)

        for key, value in removed_min:
            assert key < minority_size
            min_tree.insert(np.uint32(key), value)
        for key, value in removed_maj:
            assert key < majority_size
            maj_tree.insert(np.uint32(key), value)
                
                
        #assert len(targets) == m#, f"{targets}, {m}"
        values=targets[:i]
        #assert len(values) == m#, f"{values}, {m}"
        minority_indicator = labels[values]

        
        # report newly formed edges
        targets_min = values[minority_indicator]
        targets_maj = values[~minority_indicator]
        for t in targets_min:
            edges[n_edges,0]=source
            edges[n_edges,1]=t
            n_edges+=1
        for t in targets_maj:
            edges[n_edges,0]=source
            edges[n_edges,1]=t
            n_edges+=1

        #print(targets_min)
        # update weights
        for node in targets_min:
            t=node2index[node]
            val = min_tree.remove(t)
            assert t < minority_size
            min_tree.insert(np.uint32(t), val+1)
        #print(targets_maj, maj_tree._tree)    
        for node in targets_maj:
            t=node2index[node]
            assert t < majority_size
            val = maj_tree.remove(t)
            maj_tree.insert(np.uint32(t), val+1)
            
        # And the new node "source" has m edges to add to the list.
        if labels[source]:
            t = np.uint32(node2index[source])
            assert t < minority_size
            val=min_tree.remove(t)
            assert val==0
            min_tree.insert(t, float(m))
        else:
            t = np.uint32(node2index[source])
            assert t < majority_size
            val=maj_tree.remove(t)
            assert val==0
            maj_tree.insert(t, float(m))
    return n_edges, n_random



def LinLogHomophilySampler(N, m, minority_fraction, homophily, epsilon=10E-15, force_minority_first=False):
    assert 0 <= homophily, "Homophily needs to be in [0,1]"
    assert 1 >= homophily, "Homophily needs to be in [0,1]"
    assert 0 <= minority_fraction, "minority_fraction needs to be in [0,1]"
    assert 1 >= minority_fraction, "minority_fraction needs to be in [0,1]"
    assert epsilon > 0, "epsilon has to be larger than zero"
    assert N > m, f'{N} {m}'
        
    minority_size = int(minority_fraction * N)

        
    labels = np.zeros(N, dtype=bool)
    labels[:minority_size]=1
    if force_minority_first:
        np.random.shuffle(labels[1:])
    else:
        np.random.shuffle(labels)
    #print(labels[0])
    # we can probably tune down the sizes of these arrays
    edges=np.empty(((N-m) * m, 2), dtype=np.uint32)
    n_edges,n_random = lin_log_homo_inner(N, m ,
                                  minority_fraction,
                                  homophily,
                                  epsilon,
                                  labels,
                                  edges)
    #print(n_edges)
    assert edges.shape[0] == n_edges
    #print(n_random)
    return edges[:n_edges,:], labels






@njit
def homo_inner_multi(N, m, minority_fraction, homophily, epsilon, labels, edges):
    """
    samples edges according to barabasi + homophily but allows multi edges
    this algorithm has strict runtime O(n + m)
    """
    n_edges = 0
    minority_size = int(minority_fraction * N)
    majority_size = N - minority_size

    n_random = 0

    min_tickets = np.zeros(2 * N * m, dtype=np.uint32)
    min_weights = 0
    maj_tickets = np.zeros(2 * N * m, dtype=np.uint32)
    maj_weights = 0
    i_min = 0
    i_maj = 0
            
    targets = np.zeros(m, dtype=np.uint32)
    source = m
    for source in range(m, N):
        #print("<<<<", source)
        min_update = 0
        maj_update = 0

        ### Figure out which labels to attach to:
        if labels[source]:# belongs to minority
            f_min = homophily
            f_maj = 1-homophily
            min_update += m
        else:
            f_min = 1-homophily
            f_maj = homophily
            maj_update += m

        for i in range(0, m):
            min_term = min_weights * f_min
            maj_term = maj_weights * f_maj
            
            #print(i, min_term, maj_term, epsilon_term)
            
            

            if maj_term == 0 and min_term==0:
                # Need to assign randomly
                j=0
                while i < len(targets):
                    x = np.random.randint(0, source)
                    if not contains(targets[:i], x):
                        targets[i] = x
                        j+=1
                        i+=1
                n_random+=j
                continue
                    
                    
                    
            r=np.random.rand() * (min_term + maj_term)
            t=np.uint32(0)
            if r <= maj_term:
                assert f_maj > 0
                index = np.random.randint(0, i_maj)
                t = maj_tickets[index]
                maj_update += 1
            else:
                assert f_min > 0
                index = np.random.randint(0, i_min)
                t = min_tickets[index]
                min_update += 1

            targets[i] = t

                
        minority_indicator = labels[targets]

        for t, label in zip(targets, minority_indicator):
            edges[n_edges,0]=source
            edges[n_edges,1]=t
            n_edges+=1

            if label:
                min_tickets[i_min] = t
                i_min += 1
            else:
                maj_tickets[i_maj] = t
                i_maj += 1

        # update weights
        min_weights += min_update
        maj_weights += maj_update

        # And the new node "source" has m edges to add to the list.
        if labels[source]:
            for i in range(m):
                min_tickets[i_min] = source
                i_min += 1
        else:
            for i in range(m):
                maj_tickets[i_maj] = source
                i_maj += 1
    return n_edges, n_random


def HomophilySamplerMulti(N, m, minority_fraction, homophily, epsilon=10E-15, force_minority_first=False):
    assert 0 <= homophily, "Homophily needs to be in [0,1]"
    assert 1 >= homophily, "Homophily needs to be in [0,1]"
    assert 0 <= minority_fraction, "minority_fraction needs to be in [0,1]"
    assert 1 >= minority_fraction, "minority_fraction needs to be in [0,1]"
    assert epsilon > 0, "epsilon has to be larger than zero"
    assert N > m, f'{N} {m}'
        
    minority_size = int(minority_fraction * N)

        
    labels = np.zeros(N, dtype=bool)
    labels[:minority_size]=1
    if force_minority_first:
        np.random.shuffle(labels[1:])
    else:
        np.random.shuffle(labels)
    #print(labels[0])
    # we can probably tune down the sizes of these arrays
    edges=np.empty(((N-m) * m, 2), dtype=np.uint32)
    n_edges,n_random = homo_inner_multi(N, m ,
                                  minority_fraction,
                                  homophily,
                                  epsilon,
                                  labels,
                                  edges)
    #print(n_edges)
    assert edges.shape[0] == n_edges
    #print(n_random)
    return edges[:n_edges,:], labels

@njit
def homo_inner_rejection(N, m, minority_fraction, homophily, epsilon, labels, edges):
    """
    Barabasi albert + homophily generator that uses rejection sampling

    Runtime in priciple is m * n but for strange 
    """
    n_edges = 0
    minority_size = int(minority_fraction * N)
    majority_size = N - minority_size

    n_random = 0

    min_tickets = np.zeros(2 * N * m, dtype=np.uint32)
    min_weights = 0
    maj_tickets = np.zeros(2 * N * m, dtype=np.uint32)
    maj_weights = 0
    i_min = 0
    i_maj = 0
            
    
    source = m


    for source in range(m, N):
        #print("<<<<", source)
        min_update = 0
        maj_update = 0

        ### Figure out which labels to attach to:
        if labels[source]:# belongs to minority
            f_min = homophily
            f_maj = 1-homophily
            min_update += m
        else:
            f_min = 1-homophily
            f_maj = homophily
            maj_update += m
        targets = np.zeros(m, dtype=np.uint32)
        n_targets = 0
        n_tries = 0
        while (n_targets < m) and (n_tries < source):
            n_tries += 1
            min_term = min_weights * f_min
            maj_term = maj_weights * f_maj
            
            #print(i, min_term, maj_term, epsilon_term)
            
            

            if maj_term == 0 and min_term==0:
                # Need to assign randomly
                j=0
                while n_targets < len(targets):
                    x = np.random.randint(0, source)
                    if not contains(targets[:n_targets], x):
                        targets[n_targets] = x
                        j+=1
                        n_targets+=1
                n_random+=j
                continue
                    
                    
                    
            r=np.random.rand() * (min_term + maj_term)
            t=np.uint32(0)
            if r <= maj_term:
                assert f_maj > 0
                index = np.random.randint(0, i_maj)
                t = maj_tickets[index]
                maj_update += 1
            else:
                assert f_min > 0
                index = np.random.randint(0, i_min)
                t = min_tickets[index]
                min_update += 1
            
            # Change to normal is here
            if not contains(targets, t):
                targets[n_targets] = t
                n_targets+=1

        targets = targets[:n_targets]
        minority_indicator = labels[targets]

        for t, label in zip(targets, minority_indicator):
            edges[n_edges,0]=source
            edges[n_edges,1]=t
            n_edges+=1

            if label:
                min_tickets[i_min] = t
                i_min += 1
            else:
                maj_tickets[i_maj] = t
                i_maj += 1

        # update weights
        min_weights += min_update
        maj_weights += maj_update

        # And the new node "source" has m edges to add to the list.
        if labels[source]:
            for i in range(m):
                min_tickets[i_min] = source
                i_min += 1
        else:
            for i in range(m):
                maj_tickets[i_maj] = source
                i_maj += 1
    return n_edges, n_random


def HomophilySamplerRejection(N, m, minority_fraction, homophily, epsilon=10E-15, force_minority_first=False):
    assert 0 <= homophily, "Homophily needs to be in [0,1]"
    assert 1 >= homophily, "Homophily needs to be in [0,1]"
    assert 0 <= minority_fraction, "minority_fraction needs to be in [0,1]"
    assert 1 >= minority_fraction, "minority_fraction needs to be in [0,1]"
    assert epsilon > 0, "epsilon has to be larger than zero"
    assert N > m, f'{N} {m}'
        
    minority_size = int(minority_fraction * N)

        
    labels = np.zeros(N, dtype=bool)
    labels[:minority_size]=1
    if force_minority_first:
        np.random.shuffle(labels[1:])
    else:
        np.random.shuffle(labels)
    #print(labels[0])
    # we can probably tune down the sizes of these arrays
    edges=np.empty(((N-m) * m, 2), dtype=np.uint32)
    n_edges,n_random = homo_inner_rejection(N, m ,
                                  minority_fraction,
                                  homophily,
                                  epsilon,
                                  labels,
                                  edges)
    #print(n_edges)
    #assert edges.shape[0] == n_edges
    #print(n_random)
    return edges[:n_edges,:], labels