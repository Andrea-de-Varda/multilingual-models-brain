#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 17:19:23 2024

@author: dev
"""
def swap_elements(a, b, indices):
    a, b = copy.deepcopy(a), copy.deepcopy(b)  # avoid changing the original arrays
    for i in indices:
        a[i], b[i] = b[i], a[i]
    return [a, b]

def generate_swaps(a1, a2):
    # Define inner function for swapping elements
    # Generate all combinations of index sets for swapping
    swap_indices = []
    for r in range(1, len(a1) + 1):  # start from 1 to exclude no swap scenario
        swap_indices.extend(itertools.combinations(range(len(a1)), r))
    result = []
    for indices in swap_indices:
        result.append(swap_elements(a1, a2, indices))

    return result

def permutation_statistic(a1, a2):
    if np.mean(a1) < np.mean(a2):
        a1, a2 = a2, a1
    mean_diff = np.mean(a1) - np.mean(a2)
    swaps = generate_swaps(a1, a2)
    diffs = [(np.mean(xa1) - np.mean(xa2)) for xa1, xa2 in swaps]
    p = sum([(diff >= mean_diff) for diff in diffs])/len(diffs) # adding the held-out one, mean diff
    return p

for model in model_names:
    p = permutation_statistic(get_median_layerwise(load(model, monol=False), colname = "r", give_all = True), get_median_layerwise(load(model, monol=False, random=True), colname = "r", give_all = True))
    print(model, p)
    
    
    
def swap_elements(a, b, indices):
    a, b = copy.deepcopy(a), copy.deepcopy(b)  # avoid changing the original arrays
    for i in indices:
        a[i], b[i] = b[i], a[i]
    return [a, b]

def generate_swaps(a1, a2):
    # Define inner function for swapping elements
    # Generate all combinations of index sets for swapping
    swap_indices = []
    for r in range(1, len(a1) + 1):  # start from 1 to exclude no swap scenario
        swap_indices.extend(itertools.combinations(range(len(a1)), r))
    result = []
    for indices in swap_indices:
        result.append(swap_elements(a1, a2, indices))

    return result

def permutation_statistic(a1, a2):
    if np.mean(a1) < np.mean(a2):
        a1, a2 = a2, a1
    mean_diff = np.mean(a1) - np.mean(a2)
    swaps = generate_swaps(a1, a2)
    diffs = [(np.mean(xa1) - np.mean(xa2)) for xa1, xa2 in swaps]
    p = sum([(diff >= mean_diff) for diff in diffs])/len(diffs) # adding the held-out one, mean diff
    return p

for model in model_names:
    p = permutation_statistic(get_best_layerwise(load(model), give_all = True), get_best_layerwise(load(model, random=True), give_all = True))
    print(model, p)
    
for model in model_names:
    p = permutation_statistic(get_median_layerwise(load(model), give_all = True), get_median_layerwise(load(model, random=True), give_all = True))
    print(model, p)
