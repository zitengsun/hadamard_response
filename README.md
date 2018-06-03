# Hadamard Response: learning distribution privately and efficiently with little communication
This package implements Hadamard Response as well as three former schemes for locally private distribution learning. We also provide a script to compare their performance on synthetic data.

For complete description and analysis of the schemes, please refer to [Communication, Efficient Sample Optimal Linear Time Locally Private Discrete Distribution Estimation](https://arxiv.org/abs/1802.04705) by [Jayadev Acharya](http://people.ece.cornell.edu/acharya/), Ziteng Sun and Huanyu Zhang and its references.


## Table of contents
* [Prerequisites](#prerequisites)
* [Brief Introduction](#brief-introduction)
* [Hadamard Response](#support-coverage-estimator)
* [Subset Selection]
* [Classical methods: Randomized Response and RAPPOR]
* [Comprehensive Testing Script]

## Prerequisites

This project is implemented in Python3, using [Numpy](http://www.numpy.org) and [matplotlib](https://matplotlib.org/index.html). Before running the code, make sure Python3, Numpy and Matplotlib are installed.

* [Instruction for installing Python3](https://docs.python.org/3/using/index.html)
* [Instruction for installing Numpy](https://www.scipy.org/install.html)
* [Instruction for installing Matplotlib](https://matplotlib.org/users/installing.html) 


## Brief Introduction


## Usage

The four schemes are implemented based on python classes. Befors using, please first import the packages and then specialize the scheme with the alphabet size and the required privacy level.

```python
    import k2k_hadamard
    import Subsetselection
    import RR_RAPPOR
    
    subset = Subsetselection.Subsetselection(k, eps) #class for subset selection algorithm
    rappor = RR_RAPPOR.RAPPOR(k, eps) #class for RAPPOR
    rr = RR_RAPPOR.Randomized_Response(k, eps) #class for Randomized Response
    hr = k2k_hadamard.Hadamard_Rand_2_modified(k, eps) #initialize hadamard response
```
When you simulate Hadamard responce on a single computer, we provide the option of encoding acceleration by storing Hadamard matrix with the expense of large memory cost. If you want to use accelerate the encoding process, just set variable *encode_acc* to be *one* when intializing hadamard response.


```python
    import k2k_hadamard
    hr = k2k_hadamard.Hadamard_Rand_2_modified(k, eps, encode_acc = 1) #initialize hadamard response
```

### Hadamard Response




Comprehensive script
---------
We provide ```main_entropy.py``` as an example script for our private estimator. In this script, we compare performance for these entropy estimators on different distributions including uniform, a distribution with two steps, Zipf(1/2), a distribution with Dirichlet-1 prior, and a distribution with Dirichlet-1/2 prior. We use RMSE (root-mean-square error) to indicate the performance of the estimator.

### Program arguments

* ```k int```: Set alphabet size. 
* ```eps float```: Set privacy parameter eps.
* ```l_degree int```: Set polynomial degree for private poly. Default *L=1.2 log k*.
* ```M_degree float```: Set the right endpoint of approximation interval for private poly. Default *M=2.0 log k*.
* ```N_degree int```: Set the threshold to apply polynomial estimator for private poly. Default *M=1.6 log k*.

For the parameters of poly estimator, we just use the default values in their code, which are *L=1.6 log k, M=3.5 log k, N=1.6 log k*. Please see [entropy](https://github.com/Albuso0/entropy) for more information.


Support coverage estimator
================
In this project, we implement our estimator for private estimation of support coverage.
This is a privatized version of the Smoothed Good-Toulmin (SGT) estimator of [Alon Orlitsky](http://alon.ucsd.edu/), [Ananda Theertha Suresh](http://theertha.info/), and [Yihong Wu](http://www.stat.yale.edu/~yw562/), from their paper [Optimal prediction of the number of unseen species](http://www.pnas.org/content/113/47/13283?sid=c704d36c-5237-4425-84e4-498dcd5151b1).
We compare the performance of the private and non-private statistics on both synthetic data and real-world data, including US Census name data and a text corpus from Shakespeare's Hamlet.

Some of our code is based off the SGT implementation of Orlitsky, Suresh, and Wu, graciously provided to us by Ananda Theertha Suresh. Specific files used are indicated in comments, and ```hamlet_total.csv``` is a reformatted version of a provided file. ```lastnames_total.csv``` is a subsampling of a file of [Frequently Occurring Surnames from the Census 2000](https://www.census.gov/topics/population/genealogy/data/2000_surnames.html).

Synthetic data 
---------
We provide ```main_synthetic.py``` as an example for our private estimator. In this script, we compare performance for the estimators on different distributions including uniform, a distribution with two steps, Zipf(1/2), a distribution with Dirichlet-1 prior, and a distribution with Dirichlet-1/2 prior. We use RMSE (root-mean-square error) to indicate the performance of the estimator.

### Program arguments
* ```k int```: Set alphabet size for the distribution.
* ```eps_index list```: Set privacy parameter for the private estimators.
* ```n int```: The number of the seen samples

Real data
---------
We provide ```main_real.py``` as an example for our private estimator. In this script, we compare performance for the estimators on real data. We use RMSE (root-mean-square error) to indicate the performance of the estimator.

### Program arguments
* ```file_name string```: The name of the histogram file. The histogram file must be in ```.csv```, which has only one column and each row is the number of samples for each species. We provide ```hamlet_total.csv``` and ```lastnames_total.csv``` as some examples.
* ```eps_index list```: Set privacy parameter for the private estimators.
