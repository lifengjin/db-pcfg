#!/usr/bin/env python3
# Classes and methods for facilitating communication between workers and server

import os

class PyzmqJob:
    SENTENCE=0
    QUIT=1
    COMPILE=2
    
    def __init__(self, msg_type, resource):
        self.type = msg_type
        self.resource = resource

class JobRequest:
    def __init__(self, request_size=1):
        self.request_size = request_size
        
class SentenceRequest(JobRequest):
    def __init__(self, resource_sig, request_size=1):
        self.resource_sig = resource_sig
        self.request_size = request_size
        
class SentenceJob:
    def __init__(self, index, ev_seq):
        self.index = index
        self.ev_seq = ev_seq

class RowRequest(JobRequest):
    def __init__(self, resource_sig):
        self.resource_sig = resource_sig
        self.request_size = 1
        
class CompileJob:
    def __init__(self, index, full_pi = False):
        self.index = index
        self.full_pi = full_pi

class CompletedJob:
    def __init__(self, job_type, result, success):
        self.job_type = job_type
        self.result = result
        self.success = success
        
class PyzmqParse:
    def __init__(self, index, state_list, log_prob, success=True):
        self.index = index
        self.state_list = state_list
        self.log_prob = log_prob
        self.success = success

class CompiledRow:
    def __init__(self, index, indices, data, indices_full, data_full):
        self.index = index
        self.indices = indices
        self.data = data
        self.indices_full = indices_full
        self.data_full = data_full

class ModelLocation:
    def __init__(self, ip_addr, file_path):
        self.ip_addr = ip_addr
        self.file_path = file_path
        

class ModelWrapper:
    
    INFINITE = 0
    HMM = 1
    COMPILE = 2
    
    ## If the model type is INFINITE we just have a Models object with all
    ## the various CPTs as component models. If we have HMM we have a tuple of
    ## a Models object and a scipy.sparse transition matrix object.
    def __init__(self, model_type, model, depth):
        self.model = model
        self.model_type = model_type
        self.depth = depth

def resource_current(ref_sig, comp_sig):
    return (ref_sig[0] == comp_sig[0] and ref_sig[1] == comp_sig[1])
    
def get_file_signature(filename):
    (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(filename)
    return (size, mtime)    

    
