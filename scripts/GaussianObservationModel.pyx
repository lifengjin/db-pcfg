# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# distutils: define_macros=CYTHON_TRACE=0
import numpy as np
cimport Indexer
cimport numpy as np
cimport PosDependentObservationModel
cimport models

cdef class GaussianObservationModel(PosDependentObservationModel.PosDependentObservationModel):
    cdef set_models(self, uhhmm_models):
        PosDependentObservationModel.PosDependentObservationModel.set_models(self, uhhmm_models)
        self.lex = uhhmm_models.lex

    ## In this version of the observation model, token is a continuous vector.
    ## Here we compute the joint probability of the observation dimensions given
    ## the different POS tags. So for each tag, we compute the product of
    ## probabilities of output of each dimension given the parameters to the
    ## gaussian distribution for that POS tag and dimension.
    ## For multinomial, we have G * W params.
    ## For gaussian we have G x 2 x D params for D-dimensional embeddings.
    ## Assume that the values inside the models.lex array are scipy.stats.norm distributions.
    cdef get_pos_probability_vector(self, token):
        maxes = self.indexer.getVariableMaxes()
        (a_max,b_max,g_max) = maxes
        token_vec = self.lex.embeddings[token]

        retVec = [-np.inf]
        for g in range(1,self.lex.shape[0]-1):
            #print("Computing probability for pos index %d" % (g))
            log_prob = 0.0
            ## Vectorizing:
            #print("From mean vector index %d : %s " % (g, str(self.lex.dist[g].mean())))
            vec_prob = self.lex.dist[g].pdf(token_vec)
            if vec_prob.min() <= 0.0:
                print("Found vector with minimum at 0: ", vec_prob)
            vec_log_prob = np.log(vec_prob).sum()
            if np.isnan(vec_log_prob.min()):
                print("Found nan in log probs: ", vec_log_prob)
                print("From mean vector: ", self.lex.dist[g].mean())
                print("Sampled probs:, ", vec_prob)

            #print("Sampled probs:, ", vec_prob)
            retVec.append(vec_log_prob)

        retVec.append(-np.inf)
        return np.array(retVec)
