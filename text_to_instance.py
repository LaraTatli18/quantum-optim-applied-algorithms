import numpy as np

def text_to_instance(filename):
    """ Loads file containing spins s_i, s_j and couplings J_ij, converts to matrix format for annealer"""
    coupling_list = []

    for line in open(filename):
        coupling_list.append([float(i) for i in line.strip().split()])

    return coupling_list