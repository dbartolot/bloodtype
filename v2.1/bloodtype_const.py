from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import ipdb


default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


class BloodType_Constants():

    btpt_to_int = {'O': 0, 'A': 1, 'B': 2, 'AB': 3}
    int_to_btpt = {0: 'O', 1: 'A', 2: 'B', 3: 'AB'}
    rfpt_to_int = {'+': 0, '-': 1}
    int_to_rfpt = {0: '+', 1: '-'}

    rf_pt = ['+', '-']
    bt_pt = ['O', 'A', 'B', 'AB']
    rf_pt = ['+', '-']
    bt_pt_colors = {
        'O': default_colors[0],
        'A': default_colors[1],
        'B': default_colors[2],
        'AB': default_colors[3]
    }
    btrf_pt = ['O+', 'A+', 'B+', 'AB+', 'O-', 'A-', 'B-', 'AB-']
    btrf_pt_colors = {
        'O+': default_colors[0],
        'A+': default_colors[1],
        'B+': default_colors[2],
        'AB+': default_colors[3],
        'O-': default_colors[4],
        'A-': default_colors[5],
        'B-': default_colors[6],
        'AB-': default_colors[7],
    }
    timesteptypes = {'d': 1 / 365, 'w': 7 / 365, 'm': 1 / 12, 'y': 1}
    fitness = {'O': 1, 'A': 1, 'B': 1, 'AB': 1}
