# Hadamard Response: Estimating Distributions Privately, Efficiently, and with Little Communication
This package implements Hadamard Response for locally private discrete distribution estimation proposed in [Communication, Efficient Sample Optimal Linear Time Locally Private Discrete Distribution Estimation](https://arxiv.org/abs/1802.04705). 

Also included are implementations of *k*-Randomized Response(RR), *k*-RAPPOR and Subset Selection(SS) schemes. 


The implementation is in Python3, and uses [Numpy](http://www.numpy.org) and [matplotlib](https://matplotlib.org/index.html). 

<!---
[distribution as well as three former schemes including . We also provide a script to compare their performance on synthetic data.
 For complete description and analysis of the schemes, please refer to [Communication, Efficient Sample Optimal Linear Time Locally Private Discrete Distribution Estimation](https://arxiv.org/abs/1802.04705) by [Jayadev Acharya](http://people.ece.cornell.edu/acharya/), Ziteng Sun and Huanyu Zhang and references therein.
--->



## Table of contents
* [Usage](#usage)
* [Acknowledgement](#acknowledgement)


<!---
## Brief Introduction
Given *n* independent samples from an unknown distribution, the task of distribution learning is to infer the underlying distribution. In local differential privacy setting, samples are distributed in multiple users, and instead of sending the original samples they get, each user send a randomized version of their sample to preserve privacy. This comes with the expense of higher sample complexity.
In high privacy regime, all former schemes require either a higher communication cost which is linear with the alphabet size *k* or a higher sample complexity which is a factor of *k* larger than the optimal. Our proposed scheme is the first to achieve optimal sample complexity and communication complexity in this regime. Moreover, the computation complexity at the server end is only *O(n+k)* while other optimal schemes require *Omega(nk)* time. A slightly generalized version is sample optimal in all parameter regimes with the same communication and computation complexity.
--->


<!---
* [Instruction for installing Python3](https://docs.python.org/3/using/index.html)
* [Instruction for installing Numpy](https://www.scipy.org/install.html)
* [Instruction for installing Matplotlib](https://matplotlib.org/users/installing.html) --->

## Usage
We provide the packages as *.py* files and we also provide the *.pynb* files we wrote originally for testing in the *Jupyter _Notebooks* folder.

The comments in the code files are enough to understand how to use the functions. Here we provide some examples which hopefully will be helpful. The four schemes are implemented based on python classes. Before using, please first import the packages and then specialize the scheme with the alphabet size (*k*) and the required privacy level (*eps*).

```python
    import k2k_hadamard
    import Subsetselection
    import RR_RAPPOR
    
    subset = Subsetselection.Subsetselection(k, eps) #class for subset selection algorithm
    rappor = RR_RAPPOR.RAPPOR(k, eps) #class for RAPPOR
    rr = RR_RAPPOR.Randomized_Response(k, eps) #class for Randomized Response
    hr = k2k_hadamard.Hadamard_Rand_general(k, eps) #initialize hadamard response
```
When you simulate Hadamard response on a single computer, we provide the option of encoding acceleration by storing Hadamard matrix with the expense of large memory cost. If you want to accelerate the encoding process, just set variable *encode_acc* to be *one* when intializing hadamard response.

```python
    import k2k_hadamard
    hr = k2k_hadamard.Hadamard_Rand_general(k, eps, encode_acc = 1) #initialize hadamard response
```

### Hadamard Response
We provide multiple classes for Hadamard response scheme. If you are focusing on the high privacy regime, i.e. the case when the privacy prarameter is small (small single digit), please use the class *Hadamard_Rand_high_priv*. If you want to test other cases, please use *Hadamard_Rand_general*, which is an improved version of *Hadamard_Rand_general_original*. The latter is the version we provide in the paper and it is easier to analyze, so we also provide it in the package.

The following script encodes the input string into its randomized version and then learns the distribution based on the output string:
```python
    out_string = hr.encode_string(in_list)
    prob_est = hr.decode_string(out_string) # estimate the original underlying distribution
```

If you only want to encode a single symbol, please use:

```python
    out_symbol = hr.encode_symbol(in_symbol)
```

When you decode the output string, you can also choose whether to use fast decoding and how to normalize the output. In general, fast decoding is preferred. For normalization, we provide two options, clip and normalize(default setting) and simplex projection ([Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application](https://arxiv.org/abs/1309.1541)).

### Randomized Response

The following script first encodes the input string into its randomized version and then learn the distribution based on the output string:
```python
    out_string = rr.encode_string(in_list)
    prob_est = rr.decode_string(out_string) # estimate the original underlying distribution
```
The same normalization options are also provided for randomized response.


### Subset Selection and RAPPOR
The implementation of these two are quite similar, so here we give Subset Selection as an example.

We have three encoding modes for *Subsetselection*, *standard*, *compress* and *light*. For *standard* mode, the following encoding functions encode the input string into a *0/1* matrix of size *(n,k)* (*n* is the number of inputs and *k* is the alphabet size) where each row is the codeword corresponding to a certain input. To decode, we simply use *decode_string* function to decode this matrix:

```python
    out_string = subset.encode_string_fast(in_list) 
    prob_est = subset.decode_string(outp_string,n) # estimate the original underlying distribution
```


The drawback of *standard* mode is that it will have a huge momory cost, so we provide the *light* mode, which encode the input string into a *k*-length string which represents the total number of *1*'s at each location. We will also return the time for the counting step as this should be added into the decoding time. For decoding, we use *decode_counts*:

```python
    counts, time = subset.encode_string_light(in_list) 
    prob_est = subset.decode_counts(counts,n) # estimate the original underlying distribution
```

For *compress mode*, we encode each input symbol into a string of locations of *1*'s and then concatenate them together to get a bigger string. This may result in less communication cost when the privacy parameter is relatively large. When decoding, we need to first get the histogram and then use *decode_counts*:

```python
    out_list = subset.encode_string_compress(in_list) #subset selection
    counts, temp = np.histogram(out_list,range(k+1)) # k is the alphabet size
    prob_est = subset.decode_counts(counts, n) # estimate the original underlying distribution
```


### Simulation

For simulation, we provide functions to get *geometric*, *uniform*, *two step*, *Zipf* and *Dirichlet* distributions.


```python
    import functions
    dist = generate_geometric_distribution(k,lbd)
```

In file *test.ipynb* (also available in *test.py*), we provide a function to compare the perfomance of the four schemes in terms of l_1 and l_2 error:

```python
    data = test(k, eps, rep, point_num, step_sz, init, dist, encode_acc = 1, encode_mode = 0)
    #Args:
    # k : alphabet size,  eps: privacy level, rep: repetition times to compute a point
    # point_num: number of points to compute
    # step_sz: distance between two sample sizes
    # init: initial sample size = init* step_sz
    # dist: underlying data distribution: choose from 'Uniform', 'Two_steps', 'Zipf', 'Dirchlet', 'Geometric'
    # encode_acc : control whether to use fast encoding for hadamard responce
    # mode: control encoding method for rappor and subset selection
```
You can customize the testing process by setting these parameters. The returned data file will contain the errors, time stamps, parameter settings and other related information. Plots for comparing l_1, l_2 error and decoding time will be generated. A *.mat* file with time stamp will also be stored when the testing process is done.

## Acknowledgement

We thank [Peter Kairouz](https://web.stanford.edu/~kairouzp/) and [Gautam Kamath](http://www.gautamkamath.com/) for valuable suggestions on improving the code. This research was supported by NSF through the grant NSF CCF-1657471.
