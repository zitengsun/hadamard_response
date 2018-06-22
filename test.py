
#get_ipython().run_line_magic('matplotlib', 'inline')
#import Ipynb_importer
#import Classic
import RR_RAPPOR
import Subsetselection
import k2k_hadamard
import timeit
import scipy.io as io
import datetime

import numpy as np
import random
import math
import matplotlib.pyplot as plt
import scipy.io as io
from functions import *


def test(k, eps, rep, point_num, step_sz, init, dist, encode_acc = 1, encode_mode = 0):
    # fix alphabet size and privacy level, get the error plot with respect to sample size
    
    #Args:
    # k : alphabet size,  eps: privacy level, rep: repetition times to compute a point
    # point_num: number of points to compute
    # step_sz: distance between two sample sizes
    # init: initial sample size = init* step_sz
    # dist: underlying data distribution: choose from 'Uniform', 'Two_steps', 'Zipf', 'Dirchlet', 'Geometric'
    
    # encode_acc : control whether to use fast encoding for hadamard responce
    #           recommended and default: 1 use fast encoding when k < 10000
    #                                    if memory use is high, disable this
    
    # mode: control encoding method for rappor and subset selection
    #       0 for standard, which is fast but memory intensive
    #       1 for light, which is relatively slow but not memory intensive
    #       2 for compress, where we compress the output of rappor and subsetselection into locations of ones.
    #       recommended and default: 0 when k <= 5000 n <= 1000000
    #                               if memory use is high, use light mode
    #       you can also create other modes by modifying the code
    print('Alphabet size:', k)
    print('Privacy level:', eps)
    
    indicies = [(init-1+i)*step_sz for i in range(1,point_num+1) ] # all the indicies
    
    subset = Subsetselection.Subsetselection(k,eps) #class for subset selection algorithm
    rappor = RR_RAPPOR.RAPPOR(k,eps) #class for RAPPOR
    rr = RR_RAPPOR.Randomized_Response(k, eps) #class for Randomized Response
    if encode_acc == 1:
        hr = k2k_hadamard.Hadamard_Rand_general(k,eps,1) #initialize hadamard response
    else:
        hr = k2k_hadamard.Hadamard_Rand_general(k,eps,0) #initialize hadamard response
    
    prob1 = generate_uniform_distribution(k)
    prob2 = generate_two_steps_distribution(k)
    prob3 = generate_Zipf_distribution(k,1.0)
    prob4 = generate_Dirichlet_distribution(k,1.0)
    prob5 = generate_geometric_distribution(k,0.8)

    prob_list = {
        'Uniform' : prob1,
        'Two_steps' : prob2,
        'Zipf' : prob3,
        'Dirchlet' : prob4,
        'Geometric' : prob5, 
        }
    #underlying distribution
    prob = prob_list[dist]
    
    # to store l1 errors for each method
    l1_1 = [0]*point_num
    l1_2 = [0]*point_num
    l1_3 = [0]*point_num
    l1_4 = [0]*point_num
    
    # to store l2 errors for each method
    l2_1 = [0]*point_num
    l2_2 = [0]*point_num
    l2_3 = [0]*point_num
    l2_4 = [0]*point_num

    # to store decodign time for each method
    t1_1 = [0]*point_num
    t1_2 = [0]*point_num
    t1_3 = [0]*point_num
    t1_4 = [0]*point_num

    for r in range(init, point_num + init):
        print('Iteration:', r)
        n = r*step_sz
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        count2_1 = 0
        count2_2 = 0
        count2_3 = 0
        count2_4 = 0
        t1 = 0
        t2 = 0
        t3 = 0
        t4 = 0
        for t in range(0,rep):
            #print(t)
            elements = range(0,k)
            in_list = np.random.choice(elements, n, p=prob) #input symbols
            
            #subset selection
            if encode_mode == 0: # standard mode
                outp_1 = subset.encode_string_fast(in_list) 
                start_time = timeit.default_timer()
                prob_est_1 = subset.decode_string(outp_1,n) # estimate the original underlying distribution
                t1 = t1 + timeit.default_timer() - start_time
            if encode_mode == 1: # light mode
                counts,time = subset.encode_string_light(in_list) #subset selection
                start_time = timeit.default_timer()
                prob_est_1 = subset.decode_counts(counts, n) # estimate the original underlying distribution
                t1 = t1 + time + timeit.default_timer() - start_time
            if encode_mode == 2: # compress mode
                out_list = subset.encode_string_compress(in_list) #subset selection
                start_time = timeit.default_timer()
                counts, temp = np.histogram(out_list,range(k+1))
                prob_est_1 = subset.decode_counts(counts, n) # estimate the original underlying distribution
                t1 = t1 + timeit.default_timer() - start_time
            count1 = count1 + np.linalg.norm([a_i - b_i for a_i, b_i in zip(prob, prob_est_1)], ord=1) 
            count2_1 = count2_1 + np.linalg.norm([a_i - b_i for a_i, b_i in zip(prob, prob_est_1)], ord=2)**2

            # k- RR
            sample = rr.encode_string(in_list) 
            start_time = timeit.default_timer()
            prob_est_2 = rr.decode_string(sample) # estimate the original underlying distribution
            t2 = t2 + timeit.default_timer() - start_time
            count2 = count2 + np.linalg.norm([a_i - b_i for a_i, b_i in zip(prob, prob_est_2)], ord=1) 
            count2_2 = count2_2 + np.linalg.norm([a_i - b_i for a_i, b_i in zip(prob, prob_est_2)], ord=2)**2
            
            #k-RAPPOR
            if encode_mode == 0: 
                sample = rappor.encode_string(in_list) 
                start_time = timeit.default_timer()
                outp_3 = np.sum(sample, axis=0)
                prob_est_3 = rappor.decode_counts(outp_3,n) # estimate the original underlying distribution
                t3 = t3 + timeit.default_timer() - start_time
            if encode_mode == 1:
                counts, time = rappor.encode_string_light(in_list)
                start_time = timeit.default_timer()
                prob_est_3 = rappor.decode_counts(counts,n) # estimate the original underlying distribution
                t3 = t3 + time + timeit.default_timer() - start_time
            if encode_mode == 2:
                out_list = rappor.encode_string_compress(in_list)
                start_time = timeit.default_timer()
                counts,temp = np.histogram(out_list,range(k+1))
                prob_est_3 = rappor.decode_counts(counts,n) # estimate the original underlying distribution
                t3 = t3 + timeit.default_timer() - start_time
            count3 = count3 + np.linalg.norm([a_i - b_i for a_i, b_i in zip(prob, prob_est_3)], ord=1) 
            count2_3 = count2_3 + np.linalg.norm([a_i - b_i for a_i, b_i in zip(prob, prob_est_3)], ord=2)**2

            #k-HR
            outp_4 = hr.encode_string(in_list) 
            start_time = timeit.default_timer()
            prob_est_4 = hr.decode_string(outp_4) # estimate the original underlying distribution
            t4 = t4 + timeit.default_timer() - start_time
            count4 = count4 + np.linalg.norm([a_i - b_i for a_i, b_i in zip(prob, prob_est_4)], ord=1) 
            count2_4 = count2_4 + np.linalg.norm([a_i - b_i for a_i, b_i in zip(prob, prob_est_4)], ord=2)**2

        l1_1[r-1] = count1/float(rep)
        l1_2[r-1] = count2/float(rep)
        l1_3[r-1] = count3/float(rep)
        l1_4[r-1] = count4/float(rep)
        
        l2_1[r-1] = count2_1/float(rep)
        l2_2[r-1] = count2_2/float(rep)
        l2_3[r-1] = count2_3/float(rep)
        l2_4[r-1] = count2_4/float(rep)
        
        t1_1[r-1] = t1/float(rep)
        t1_2[r-1] = t2/float(rep)
        t1_3[r-1] = t3/float(rep)
        t1_4[r-1] = t4/float(rep)
        
    plt.figure()
    plt.plot(indicies,l1_1, label = 'subset')
    plt.plot(indicies,l1_2, label = 'rr')
    plt.plot(indicies,l1_3, label = 'rappor')
    plt.plot(indicies,l1_4, label = 'hr')
    plt.legend()
    
    plt.figure()
    plt.plot(indicies,l2_1, label = 'subset')
    plt.plot(indicies,l2_2, label = 'rr')
    plt.plot(indicies,l2_3, label = 'rappor')
    plt.plot(indicies,l2_4, label = 'hr')
    plt.legend()
    
    plt.figure()
    plt.plot(indicies,t1_1, label = 'subset')
    plt.plot(indicies,t1_2, label = 'rr')
    plt.plot(indicies,t1_3, label = 'rappor')
    plt.plot(indicies,t1_4, label = 'hr')
    plt.legend()
    time = datetime.datetime.now().strftime("%m_%d_%H_%M")

    #save all the data into a mat file with time stamp
    data = {
        'time' : time,
        'absz' : k,
        'privacy' : eps,
        'repetition' : rep,
        'indices' : indicies, # indices of each point (number of samples)
        'subset_error': l1_1, #l1 error for each point
        'rr_error': l1_2,
        'rappor_error': l1_3,
        'hr_error': l1_4,
        'subset_error_l2': l2_1, #l2 error for each point
        'rr_error_l2': l2_2,
        'rappor_error_l2': l2_3,
        'hr_error_l2': l2_4,
        'subset_time': t1_1, #decoding time for each point
        'rr_time': t1_2,
        'rappor_time': t1_3,
        'hr_time': t1_4,
        'prob': prob,
        'dist': dist
    }
    para = 'k_{}_eps_{}_'.format(k,eps)
    filename = 'Data/data_' + dist + '_' + para + time
    io.savemat(filename,data)
    return data


#Testing script for comparison
k = 100 #absz
eps = 2 # privacy_para
rep = 1 #repetition time for each point
points = 10 # total number of points
step_sz = 50000 # step size between two points
init = 1 #initial step

for dist in ['Geometric']:
    print(dist)
    for eps in [2]:
        print(datetime.datetime.now())
        data = test(k,eps,rep,points,step_sz,init,dist,0,0)

