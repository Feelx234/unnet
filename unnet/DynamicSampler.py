import numpy as np
uint32_max = np.iinfo(np.uint32).max

from numba.experimental import jitclass
from numba.typed import List
from numba import uint8, uint32, float32, boolean, njit



@njit
def get_left(i):
    return int(2 * i + 1)
@njit
def get_right(i):
    return int(2 * i + 2)
@njit
def get_parent(i):
    return (i - 1) // 2 if i > 0 else 0

spec = [
    ('_items', uint32[:]), 
    ('_ipos', uint32[:]),
    ('_tree', float32[:]),
    ('_idx', uint32[:]),
    ('_back', uint32),
    ('_free', uint32[:]),
    ('_valid', uint8[:]),
    ('_n_items', uint32),
    ('_n_free', uint32),
]
@jitclass(spec)
class DynamicSampler:
    """ The dynamic sampler uses a tree structure to allow for O(log(n)) operations
    The code was converted to pure python from 
    https://git.skewed.de/count0/graph-tool/-/blob/master/src/graph/generation/dynamic_sampler.hh
    """
    def __init__(self, N):  
        #vector<Value>  self._items
        #vector<size_t> self._ipos  // position of the item in the tree

        #vector<double> self._tree  // tree nodes with weight sums
        #vector<size_t> self._idx   // index in self._items
        #int self._back             // last item in tree

        #vector<size_t> self._free  // empty leafs
        #vector<bool> self._valid   // non-removed items
        #size_t self._n_items

        self._items = np.zeros(N, dtype=np.uint32)
        self._ipos = np.zeros(N, dtype=np.uint32)

        self._tree = np.zeros(2*N, dtype=np.float32)
        self._idx = np.zeros(2*N, dtype=np.uint32)
        self._back = int(0)

        self._free = np.zeros(10, dtype=np.uint32)
        self._n_free = 0
        self._valid = np.zeros(N, dtype=np.uint8)
        self._n_items = 0




    def sample(self):
        u = np.random.rand() * self._tree[0]
        return self.find_for_sample(u)

    def find_for_sample(self, u):
        c = 0
        pos = 0
        while (self._idx[pos] == uint32_max):
            l = get_left(pos)
            a = self._tree[l]
            if (u < (a + c)):
                pos = l
            else:
                pos = get_right(pos)
                c += a
        
        i = self._idx[pos]
        return self._items[i]

    def insert(self, v, w):
        if (self._n_free==0):
            if (self._back > 0):

                # move parent to left leaf
                pos = get_parent(self._back)
                l = get_left(pos)
                self._idx[l] = self._idx[pos]
                self._ipos[self._idx[l]] = l
                self._tree[l] = self._tree[pos]
                self._idx[pos] = uint32_max

                # position new item to the right
                self._back = get_right(pos)


            pos = self._back
            self._check_size(pos)

            self._idx[pos] = self._n_items
            self._tree[pos] = w
            # changes
            #self._check_size2(self._n_items)
            self._items[self._n_items] = v
            self._valid[self._n_items] = True
            self._ipos[self._n_items] =  pos
            # end_changes
            
            self._back+=1
            self._check_size(self._back)

        else:
            pos = self._free[self._n_free-1] #changed

            i = self._idx[pos]
            self._items[i] = v
            self._valid[i] = True
            self._tree[pos] = w
            self._n_free-=1
            #self._free=self._free[:-1] #changed


        self._insert_leaf_prob(pos)
        self._n_items+=1
        return self._idx[pos]


    def remove(self, i):
        pos = self._ipos[i]
        w = self._tree[pos]#changed
        self._remove_leaf_prob(pos)
        
        if self._n_free < len(self._free):
            self._free[self._n_free] = pos
            self._n_free+=1
        else:
            new_free = np.zeros(len(self._free)+1, dtype=self._free.dtype)
            new_free[-1] = pos
            new_free[0:len(self._free)]=self._free
            self._n_free+=1
            self._free = new_free
        
        #self._free.append(pos)
        self._items[i] = 0.0
        self._valid[i] = False
        self._n_items-=1
        return w #changed
        


    def _clear(self, shrink=False):
        raise NotImplementedError
        #self._items.clear()
        #self._ipos.clear()
        #self._tree.clear()
        #self._idx.clear()
        #self._free.clear()
        #self._valid.clear()
        #if (shrink):
        #    self._items.shrink_to_fit()
        #    self._ipos.shrink_to_fit()
        #    self._tree.shrink_to_fit()
        #    self._idx.shrink_to_fit()
        #    self._free.shrink_to_fit()
        #    self._valid.shrink_to_fit()
        #self._back = 0
        #self._n_items = 0

    def _rebuild(self):
        raise NotImplementedError
        #items=[] #needs change
        #probs=[] #needs change

        #for i in range(len(self._tree)):

        #    if (self._idx[i] == uint32_max):
        #        continue
        #    j = self._idx[i]
        #    if (not self._valid[j]):
        #        continue
        #    items.append(self._items[j])
        #    probs.append(self._tree[i])


        #self._clear(True)

        #for i in range(len(items)):
        #    self.insert(items[i], probs[i])
    def _check_size2(self, i):
        if (i >= len(self._idx)):
            raise NotImplementedError
            #changes
            new_idx = np.full(i + 1, uint32_max, dtype=self._idx.dtype)
            new_idx[0:len(self._idx)]=self._idx
            self._idx = new_idx
            #self._idx.resize(i + 1, )
            
            new_tree = np.zeros(i + 1, dtype=self._tree.dtype)
            new_tree[0:len(self._tree)]=self._tree
            self._tree = new_tree

    def _check_size(self, i):
        if (i >= len(self._tree)):
            #changes
            new_idx = np.full(i + 1, uint32_max, dtype=self._idx.dtype)
            new_idx[0:len(self._idx)]=self._idx
            self._idx = new_idx
            #self._idx.resize(i + 1, )
            
            new_tree = np.zeros(i + 1, dtype=self._tree.dtype)
            new_tree[0:len(self._tree)]=self._tree
            self._tree = new_tree
            #self._tree.resize(i + 1, 0)

    def _remove_leaf_prob(self, i):
        parent = i
        w = self._tree[i]
        while (parent > 0):
            parent = get_parent(parent)
            self._tree[parent] -= w
        self._tree[i] = 0

    def _insert_leaf_prob(self, i):
        parent = i
        w = self._tree[i]

        while (parent > 0):
            parent = get_parent(parent)
            self._tree[parent] += w
