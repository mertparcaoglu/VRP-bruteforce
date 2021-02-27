# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 09:53:40 2021

@author: Parcaoglu
"""
import operator
#import time
from json import load, dump
from itertools import permutations, product
from itertools import combinations_with_replacement as cwr
import numpy as np


class Nested_Pass(Exception): pass


# If any route among a set of jobs takes a longer time than the current best route,
# the algorithm should continue with the next set, and exit the loop of this set.


def read_data(path):
    """
    Reads data, and returns required information from data.
    :param path: The path of data
    vhc_: Vehicles info.
    jobs_: Jobs info.
    matrix_: Distance matrix between locations (in seconds)
    numbers: # of vehicles and jobs respectively
    :return: dict, dict, np.array 2D, Tuple(int, int)
    """
    with open(path, 'r') as f:
        data_ = load(f)

    vhc_, jobs_, matrix_ = data_['vehicles'], data_['jobs'], np.array(data_['matrix'])
    numbers = (len(vhc_), len(jobs_))
    return vhc_, jobs_, matrix_, numbers


def slicer(dist):
    """
    Slice function generator according to given distributions
    :param dist: Numerical distribution of jobs
    """
    a = 0
    for d in dist:
        yield slice(a, a + d)
        a += d


def find_jobs_id(arr, loc):
    """
    Fetches job id if from location index
    :param arr: array of jobs
    :param loc: location index
    :return: job id
    """
    for a_ in arr:
        if a_['location_index'] == loc:
            return a_['id']
    return 'Wrong Value!'


def find_vhc_id(arr, loc):
    """
    Fetches vehicle id if from start index
    :param arr: array of vehicles
    :param loc: start index
    :return: vehicle id
    """
    for a_ in arr:
        if a_['start_index'] == loc:
            return a_['id']
    return 'Wrong Value!'


def brute_force_routing(vhc_indices, jobs_indices, mtx, nbr_):
    """
    Minimizes the maximum of vehicle route durations using breaks on brute force algorithm.
    Before starting the algorithm, best values and the minimum threshold value are generated. Min. threshold is equal to
    the minimum distance between any vehicle and any job (i.e. minimum start duration).
    Algorithm has two main tricks to speed up searching the best routes.
    Firstly, frames of job slices are generated like (3,2,2),(4,3,0), these frames are sorted in tuples as desc.
    Then, these slices are sorted by looking last value of the tuple, because longer durations are expected from tuples
    that have idle vehicles.
    Second trick is that nested loops are broken by nested_break if route length without vehicle + min. th is greater or
    equal to current best value.
    :param vhc_indices: array
    :param jobs_indices: array
    :param mtx: Distance matrix
    :param nbr_: Tuple(# of vehicles, # of jobs)
    :return: Best routes, vehicles, best total duration
    """
    best, best_rt, best_c, th = 1e100, [], [], np.min(mtx[:, jobs_indices][vhc_indices])
    slices = [sorted(k, reverse=True) for k in cwr(range(nbr_[1] + 1), nbr_[0]) if sum(k) == nbr_[1]]
    slices.sort(reverse=True, key=operator.itemgetter(-1))
    for distribution, per_ in product(slices, permutations(jobs_indices)):
        try:
            rt = []
            for sl_ in slicer(distribution):
                seq = per_[sl_]
                inner_score = mtx[seq[:-1], seq[1:]].sum()
                if inner_score + th >= best:
                    raise Nested_Pass()
                rt.append([inner_score, seq])
            for c in permutations(vhc_indices):
                score = max([mtx[c_, r_[1][0]] + r_[0] for c_, r_ in zip(c, rt) if len(r_[1]) > 0])
                if best > score:
                    best, best_rt, best_c = score, rt, c
        except Nested_Pass:
            pass
    return best_rt, best_c, best


if __name__ == '__main__':
    #basla = time.time()
    vhc, jobs, matrix, n_ = read_data('getir_algo_input.json')
    vhc_ind, jobs_ind = [veh['start_index'] for veh in vhc], [job['location_index'] for job in jobs]

    besties = brute_force_routing(vhc_ind, jobs_ind, matrix, n_)
    results = {'Total Delivery Duration': str(besties[2]),
               'Vehicle Routes': {str(find_vhc_id(vhc, k)): [str(find_jobs_id(jobs, v_)) for v_ in v[1]]
                                  for k, v in zip(besties[1], besties[0])}}
    #print(results)
    with open('getir_algo_output.json', 'w') as json_file:
        dump(results, json_file)
    #print(time.time() - basla)
