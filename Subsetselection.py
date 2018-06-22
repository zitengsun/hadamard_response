
# coding: utf-8

# In[18]:


#%matplotlib inline
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import timeit
import itertools
from functions import *
#from scipy.special import comb
#from scipy import stats


class Subsetselection:
    def __init__(self, absz, pri_para): # absz: alphabet size, pri_para: privacy parameter
        self.insz = absz #input alphabet size k
        self.exp = math.exp(pri_para)
        self.d = int(math.ceil(1.0*self.insz/(self.exp+1))) # number of 1s in output bit string 
        self.p = (1.0*self.d*self.exp)/(self.d*self.exp+self.insz-self.d)
        self.q = 1.0*(self.d-self.p)/(self.insz-1)
        
    def encode_symbol(self,ori):  # encode a single symbol into a privatized version
        #faithful implementation of the subset selection protocal from the following paper
        # https://arxiv.org/abs/1702.00610
        # You can also refer to the appendix of https://arxiv.org/abs/1802.04705 for a complete description
        sample = np.zeros(self.insz, dtype='bool')
        y_i = np.random.binomial(1,self.p)   
        if(y_i):
            sample[ori]=True
            temp = np.delete(np.array(range(self.insz)),ori, 0)
            temp2= np.random.choice(temp,self.d-1,replace=False)
            sample [temp2]= True
        else:
            sample[ori]=False
            temp = np.delete(np.array(range(self.insz)),ori, 0)
            temp2= np.random.choice(temp,self.d,replace=False)
            sample [temp2]= True
        return sample
    
    def encode_string(self,in_list):  # encode string into a privatized string
        out_list = [self.encode_symbol(x) for x in in_list]
        return out_list
    
    def encode_string_fast(self,in_list):  # encode string into a privatized string
        # a fast implementation of the subset selection protocal
        #instead of just selecting exactlly d ones, set each bit to be one independently with true expectation
        n = len(in_list)
        out_list = np.zeros((n, self.insz))
        flip = np.random.random_sample((n, self.insz))
        
        for i in range(n):
            out_list[i,in_list[i]] = np.logical_or(0,flip[i,in_list[i]] <self.p)
        return np.logical_or(out_list, flip < self.q)

    def encode_string_light(self,in_list):  # encode string into a privatized string
        #a light, fast implementation of the former method which is less memory intensive
        n = len(in_list)
        counts = np.zeros(self.insz)
        time = 0
        for i in range(n):
            private_samples_subset = np.zeros(self.insz)
            flip = np.random.random_sample(self.insz)
            private_samples_subset[in_list[i]] = (flip[in_list[i]] < self.p)
            private_samples_subset = np.logical_or(private_samples_subset, flip < self.q)
            start_time = timeit.default_timer()
            counts = counts +  private_samples_subset
            time = time + timeit.default_timer() - start_time
        return counts, time
    
    def encode_string_compress(self,in_list):  # encode string into a privatized string
        #compress the output of subset selection into a list of locations of ones.
        n = len(in_list)
        out = [0]*n
        for i in range(n):
            private_samples_subset = np.zeros(self.insz)
            flip = np.random.random_sample(self.insz)
            private_samples_subset[in_list[i]] = (flip[in_list[i]] < self.p)
            private_samples_subset = np.logical_or(private_samples_subset, flip < self.q)
            out[i] = np.where(private_samples_subset)[0]
        #print(out)
        out_list = np.concatenate(out)
        return out_list

    def decode_counts(self, counts, length, normalization = 0): # get the privatized string and learn the original distribution
        #input is the hisogram of each location
        temp1 = ((self.insz-1)*self.exp+1.0*(self.insz-1)*(self.insz-self.d)/self.d) / ((self.insz-self.d)*(self.exp-1))
        temp2 = ((self.d-1)*self.exp+self.insz-self.d) / (1.0*(self.insz-self.d)*(self.exp-1))
        #print (np.sum(out_list, axis=0))
        p_estimate = (1.0*counts*temp1/length)-temp2
        if normalization == 0: 
            dist = probability_normalize(p_estimate) #clip and normalize
        if normalization == 1:
            dist = project_probability_simplex(p_estimate) #simplex projection
            
        #p_estimate = np.maximum(p_estimate,0)
        #norm = np.sum(p_estimate)
        #p_estimate = p_estimate/float(norm)
        return dist
    
    def decode_string(self, out_list, length, normalization = 0): # get the privatized string and learn the original distribution
        #input is the matrix consisting of all the bit vectors
        temp1 = ((self.insz-1)*self.exp+1.0*(self.insz-1)*(self.insz-self.d)/self.d) / ((self.insz-self.d)*(self.exp-1))
        temp2 = ((self.d-1)*self.exp+self.insz-self.d) / (1.0*(self.insz-self.d)*(self.exp-1))
        #print (np.sum(out_list, axis=0))
        p_estimate = (1.0*np.sum(out_list, axis=0)*temp1/length)-temp2
        if normalization == 0: 
            dist = probability_normalize(p_estimate) #clip and normalize
        if normalization == 1:
            dist = project_probability_simplex(p_estimate) #simplex projection
        
        #p_estimate = np.maximum(p_estimate,0)
        #norm = np.sum(p_estimate)
        #p_estimate = p_estimate/float(norm)
        return dist
