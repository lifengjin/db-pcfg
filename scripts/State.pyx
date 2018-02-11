#!python3

import numpy as np

def sentence_string(state_list):
    return str(list(map(lambda x: x.str(), state_list)))

# The set of random variable values at one word
# There will be one of these for every word in the training set
cdef class State:
    def __init__(self, int depth, state=None):
        self.depth = depth
        
        if state is None:
            self.f = -1
            self.j = -1
            self.a = np.zeros(depth, dtype=int)
            self.b = np.zeros(depth, dtype=int)
            self.g = 0
        else:
            (self.f, self.j, self.a, self.b, self.g) = state.f, state.j, state.a, state.b, state.g

    # def str(self):
    #     string = ''
    #
    #     f_str = '+/' if self.f==1 else '-/'
    #     string += f_str
    #     j_str = '+' if self.j==1 else '-'
    #     string += j_str
    #
    #     string += "::"
    #
    #     for d in range(0, self.depth):
    #
    #         if self.a[d] > 0 or d == 0:
    #             if d > 0:
    #                 string += ";"
    #             string += "ACT" + str(self.a[d]) + '/AWA' + str(self.b[d])
    #
    #     string += ':POS' + str(self.g)
    #
    #     return string

    def str(self):  # ordering F J A B P
        string = ''

        f_str = '+::' if self.f == 1 else '-::'
        string += f_str
        j_str = '+' if self.j == 1 else '-'
        string += j_str

        string += "::"

        for d in range(0, self.depth):

            if self.a[d] > 0 or d == 0:
                if d > 0:
                    string += ";"
                string += "ACT" + str(self.a[d]) + '/AWA' + str(self.b[d])

        string += '::POS' + str(self.g)
        return string

    def raw_str(self):  # ordering J A B F P
        string = ''


        j_str = '+' if self.j == 1 else '-'
        string += j_str

        string += "::"

        for d in range(0, self.depth):

            if self.a[d] > 0 or d == 0:
                if d > 0:
                    string += ";"
                string += "ACT" + str(self.a[d]) + '/AWA' + str(self.b[d])
        f_str = '::+' if self.f == 1 else '::-'
        string += f_str
        string += '::POS' + str(self.g)
        return string

    def unfiltered_str(self):  # ordering J A B F P, but showing the whole stack
        string = []

        j_str = 'J+' if self.j == 1 else 'J-'
        string.append(j_str)

        for d in range(0, self.depth):
            string.append("ACT" + str(self.a[d]) + '/AWA' + str(self.b[d]))
        string.append('F+') if self.f == 1 else string.append('F-')
        string.append('POS' + str(self.g))
        return '::'.join(string)


    def to_list(self):
        return (self.f, self.j, self.a, self.b, self.g)

    def max_fork_depth(self):
        return self.max_awa_depth()

    def max_awa_depth(self):
        ## Essentially empty -- used for first time step
        if self.b[0] == 0:
            return -1
        
        ## If we encounter a zero at position 1, then the depth is 0
        for d in range(1, self.depth):
            if self.b[d] == 0:
                return d-1
        
        ## Stack is full -- if d=4 then max depth index is 3
        return self.depth-1

    def max_awa_depth_err(self):
        ## Essentially empty -- used for first time step
        print("A vector is: %s" % ( '::'.join(str(x) for x in self.a) ) )
        print("B vector is: %s" % ( '::'.join(str(x) for x in self.b) ) )
        
        if self.b[0] == 0:
            print("B is all zeros so I'm returning")
            return -1
        
        ## If we encounter a zero at position 1, then the depth is 0
        for d in range(1, self.depth):
            print("Checking b at depth %d" % ( d ) )
            if self.b[d] == 0:
                print("b[%d] = 0 so i'm returning" % d)
                return d-1
        
        ## Stack is full -- if d=4 then max depth index is 3
        print("Did not find any zeros so returrning %d" % (self.depth-1) )
        return self.depth-1

    def max_act_depth(self):
        ## Essentially empty -- used for first time step
        if self.a[0] == 0:
            return -1
        
        ## If we encounter a zero at position 1, then the depth is 0
        for d in range(1, self.depth):
            if self.a[d] == 0:
                return d-1
        
        ## Stack is full -- if d=4 then max depth index is 3
        return self.depth-1

    def depth_check(self):
        if not self.max_awa_depth() == self.max_act_depth():
            print("ERROR: awa depth for this state is different than act depth")
            print("Act=%s, awa=%s" % ('::'.join(str(x) for x in self.a), '::'.join(str(x) for x in self.b) ) )
            return False
        
        return True
                
    def __reduce__(self):
        d = {}
        d['depth'] = self.depth
        d['f'] = self.f
        d['j'] = self.j
        d['a'] = self.a
        d['b'] = self.b
        d['g'] = self.g
        return (State, (self.depth,), d)
    
    def __setstate__(self, d):
        self.depth = d['depth']
        self.f = d['f']
        self.j = d['j']
        self.a = d['a']
        self.b = d['b']
        self.g = d['g']
