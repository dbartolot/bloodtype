import numpy as np
from matplotlib import pyplot as plt
import random
from person import Person
import pickle
import os
import pandas as pd

import ipdb

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


class BloodType:

    states = []
    population = []
    populationsize = 0
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
    timestep = 7 / 365
    timesteptype = 'w'
    time = 0
    timesteptypes = {'d': 1 / 365, 'w': 7 / 365, 'm': 1 / 12, 'y': 1}
    deathRate = 0.0086
    deaths = 0
    birthRate = 0.0094
    births = 0
    fitness = {'O': 1, 'A': 1, 'B': 1, 'AB': 1}

    def __init__(self):
        pass

    def steps(self):
        pass

    def step(self):
        pass

    def update_time(self):
        pass

    def update_age(self):
        pass

    def births(self):
        pass

    def deaths(self):
        pass

    def log_states(self):
        pass

    def save_states(self):
        pass

    def generate_populations(self):
        pass

    def update_fitness(self, fitness):
        pass
