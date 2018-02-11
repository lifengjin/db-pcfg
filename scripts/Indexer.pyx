# cython: profile=True
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE=1
cimport cython
import State
import numpy as np
cimport numpy as np

cdef class Indexer:
    #cdef int depth, fj_size, a_max, b_max, g_max, state_size, a_size, b_size
    def __init__(self, models):
        self.depth = len(models.F)
        self.f_size = 2
        self.j_size = 2
        self.fj_size = 4
        self.a_max = models.A[0].dist.shape[-1]
        self.b_max = models.B_J0[0].dist.shape[-1]
        self.g_max = models.pos.dist.shape[-1]
        self.state_size = self.fj_size * ((self.a_max * self.b_max) ** self.depth) * self.g_max
        self.a_size = self.a_max**self.depth
        self.b_size = self.b_max**self.depth
        self.EOS_index = 0

    def getVariableMaxes(self):
        return (self.a_max, self.b_max, self.g_max)

    def get_state_size(self):
        return self.state_size

    def get_EOS(self):
        assert self.EOS_index % self.g_max == 0, "EOS index mod g is not zero!"
        return self.EOS_index / self.g_max

    def get_EOS_full(self):
        if not self.EOS_index:
            EOS_f = 0
            EOS_j = 1
            EOS_a = np.zeros(self.depth, dtype=int)
            EOS_b = np.zeros(self.depth, dtype=int)
            EOS_g = 0
            self.EOS_index = self.getStateIndex(EOS_j, EOS_a, EOS_b, EOS_f, EOS_g)
        return self.EOS_index

    # def get_EOS_1wrd(self): #?
    #     return int(3*self.state_size / (4*self.g_max))
    #
    # def get_EOS_1wrd_full(self): #?
    #     if not self.EOS_1wrd_index:
    #         EOS_f = 0
    #         EOS_j = 1
    #         EOS_a = np.zeros(self.depth)
    #         EOS_b = np.zeros(self.depth)
    #         EOS_g = 0
    #         self.EOS_index = self.getStateIndex(EOS_j, EOS_a, EOS_b, EOS_f, EOS_g)
    #     return self.EOS_index
    #     return int(3*self.state_size / 4)

    #@profile
    def extractState(self, int index):
        (j, aStack, bStack, f, g) = self.extractStacks(index)
        state = State.State(self.depth)
        state.f = f
        state.j = j
        state.a = aStack
        state.b = bStack
        state.g = g
        return state

    cpdef public int extractPos(self, int index):
        cdef int j_ind, a_ind, b_ind, g, f_ind

        (j_ind, a_ind, b_ind, f_ind, g) = np.unravel_index(index, (self.j_size, self.a_size, self.b_size,
                                                                   self.f_size, self.g_max))
        return g

    #@profile
    cpdef public extractStacks(self, int index):
        cdef int  a_ind, b_ind, g, f, j
        cdef np.ndarray a, b

        (j_ind, a_ind, b_ind, f_ind, g) = np.unravel_index(index, (self.j_size, self.a_size, self.b_size,
                                                                   self.f_size, self.g_max))
        f = f_ind
        j = j_ind

        a = np.array(np.unravel_index(a_ind, [self.a_max] * self.depth), dtype=int)
        b = np.array(np.unravel_index(b_ind, [self.b_max] * self.depth), dtype=int)

        return j, a, b, f, g


    def extractAct(self, index):
        (j, aStack, bStack,f, g) = self.extractStacks(index)
        for d in range(self.depth - 1, -1, -1):
            if aStack[d] > 0:
                return aStack[d]
        else:
            return 0


    def extractAwa(self, index):
        ( j, aStack, bStack, f, g) = self.extractStacks(index)
        for d in range(self.depth - 1, -1, -1):
            if bStack[d] > 0:
                return bStack[d]
        else:
            return 0

    ## We compose a state index from separate fj, a, b, and g indexes.
    ## Things get a little trickier at d == D because at time t=1 we technically have
    ## an expansion at d=-1, meaning the fj stack is completely empty. (At d=1 we just
    ## fudged it and had the expansion at d=0. The result is we have to check whether the
    ## depth is -1 and if so we have a special case, since there's only one place where it
    ## would happen and we don't want to blow up the state space.
    cpdef int getStateIndex(self, int j, np.ndarray a, np.ndarray b, int f, int g):
        cdef int d, index
        # print(type(j), j, type(a), a, type(b), b, type(f), f, type(g), g)
        a_stack  = np.ravel_multi_index(a, [self.a_max] * self.depth)
        b_stack = np.ravel_multi_index(b, [self.b_max] * self.depth)

        index = np.ravel_multi_index((j, a_stack, b_stack, f, g), (self.j_size, self.a_size, self.b_size,
                                                                   self.f_size, self.g_max))

        return index

    # cpdef getStateTuple(self, int f, int j, np.ndarray a, np.ndarray b):
    #     cdef int start
    #     start = self.getStateIndex(f,j,a,b,0)
    #     return (start,start+self.g_max-1)

    # def get_g_vector(self):
    #     f = np.zeros(self.state_size)
    #     for i in range(0, self.state_size):
    #          f[i] = i % self.g_max
    #
    #     return f
