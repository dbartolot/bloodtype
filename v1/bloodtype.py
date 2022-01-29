import numpy as np
from matplotlib import pyplot as plt
import random
from person import Person
import os
import pandas as pd
import time
from tqdm import trange
import math

default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf', '#1a55FF']

SCALER = 1.4
DEATH_RATE = 0.0095*SCALER
BIRTH_RATE = 0.0096*SCALER


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
    death_rate = 0.01
    deaths = 0
    birth_rate = 0.011
    births = 0
    fitness = {
        'O': 1,
        'A': 1,
        'B': 1,
        'AB': 1
    }
    filename = 'model.pkl'
    plots_dir = './plots/'
    log_file = "log.txt"

    def __init__(self,
                 startsize,
                 timesteptype='w',
                 death_rate=0.01,
                 birth_rate=0.013,
                 filename='model.pkl'):
        self.states = []
        self.population = []

        self.timestep = 7 / 365
        self.timesteptype = 'w'
        self.time = 0
        self.deaths = 0
        self.births = 0

        self.death_rate = death_rate
        self.birth_rate = birth_rate

        self.check_directory(self.plots_dir)
        t = time.localtime()
        self.plots_dir = './plots/' + time.strftime('%Y-%b-%d_%H%M', t) + '/'

        self.setTimesteps(timesteptype)

        # plt.ion()
        # self.fig, self.ax = plt.subplots(nrows=1, ncols=2)

        self.populationsize = startsize
        self.population = [Person() for i in range(self.populationsize)]
        self.filename = filename

    def set_fitness(self, fitness):
        self.fitness = fitness
        self.log("#{}#: Fitness = {}\n".format(self.time, self.fitness))

    def setTimesteps(self, timesteptype):
        self.timesteptype = timesteptype
        self.timestep = self.timesteptypes[self.timesteptype]
        self.log("#{}#: timestep type = {}\n".format(
            self.time, self.timesteptype))

    def set_death_rate(self, value):
        self.death_rate = value
        self.log("#{}#: death rate = {}\n".format(self.time, self.death_rate))

    def get_death_rate(self):
        return self.death_rate * self.timestep

    def set_birth_rate(self, value):
        self.birth_rate = value
        self.log("#{}#: birth rate = {}\n".format(self.time, self.birth_rate))

    def get_birth_rate(self):
        return self.birth_rate * self.timestep

    def step(self, bt_mutation=None, rf_mutation=None, mutations=0):

        self.killPersons()
        self.generateOffsprings(bt_mutation=bt_mutation,
                                rf_mutation=rf_mutation,
                                mutations=mutations)
        self.time += self.timestep

        self.logState()
        # update plots
        # self.updateSize()

    def steps(self, steps, bt_mutation=None, rf_mutation=None,
              mutations=None, print_state=True):
        t = trange(steps)
        for i in t:
            t.set_description("Population Size {}, year {:.1f}".format(
                self.populationsize, self.time))
            t.refresh()
            self.step(bt_mutation=bt_mutation,
                      rf_mutation=rf_mutation,
                      mutations=mutations)

        if print_state:
            self.print_state()

    def get_deathlist(self):
        cdf = np.cumsum([self.fitness[p.bt_pt] for p in self.population])
        cdf = cdf / cdf[-1]
        self.deaths = round(self.populationsize * self.get_death_rate())

        deathlist = [None] * self.deaths
        for i in range(self.deaths):
            while True:
                pos = np.sum(cdf - random.random() < 0)
                if pos not in deathlist:
                    deathlist[i] = pos
                    break
        return deathlist

    def killPersons(self):
        deathlist = self.get_deathlist()
        deathlist.sort(reverse=True)
        if deathlist is not None:
            for i in deathlist:
                self.removePerson(i)

    def removePerson(self, i):
        self.population.pop(i)
        self.populationsize = self.populationsize - 1

    def generateOffsprings(self,
                           bt_mutation=None,
                           rf_mutation=None,
                           mutations=0):

        self.births = self.populationsize * self.get_birth_rate()
        if random.random() < self.births-int(self.births):
            self.births = math.ceil(self.births)
        else:
            self.births = round(self.births)

        if bt_mutation not in self.bt_pt and \
                rf_mutation not in self.rf_pt:
            mutations = 0

        # reduce number of mutations to the proper size of births
        if (mutations is None) or (self.births < mutations):
            mutations = self.births

        for i in range(mutations):
            i, j = self.choseParents()
            child = self.population[i].genMutatedOffspring(
                self.population[j],
                bt_gtMutation=bt_mutation,
                rf_gtMutation=rf_mutation)
            self.population.append(child)
            self.populationsize = self.populationsize + 1

        for i in range(self.births - mutations):
            i, j = self.choseParents()
            child = self.population[i].genOffspring(self.population[j])
            self.population.append(child)
            self.populationsize = self.populationsize + 1

    def generateOffspring(self):
        i = random.randint(0, self.populationsize - 1)
        sexs = np.array([p.sex for p in self.population])
        j = np.random.choice(np.arange(self.populationsize)[
                             sexs != self.population[i].sex], 1)[0]
        child = self.population[i].genOffspring(self.population[j])
        return child

    def choseParents(self):
        i = random.randint(0, self.populationsize - 1)
        sexs = np.array([p.sex for p in self.population])
        j = np.random.choice(np.arange(self.populationsize)[
                             sexs != self.population[i].sex], 1)[0]
        return (i, j)

    def logState(self):
        bt_ptFromPopulation = [p.bt_pt for p in self.population]
        bt_ptCounts = [bt_ptFromPopulation.count(
            p) for p in self.bt_pt]

        btrf_ptFromPopulation = [p.bt_pt + p.rf_pt for p in self.population]
        btrf_ptCounts = [btrf_ptFromPopulation.count(
            p) for p in self.btrf_pt]
        sexs = [np.sum([i.sex == "f" for i in self.population]), 0]
        sexs[1] = self.populationsize-sexs[0]
        currentstate = [self.time,
                        len(self.population),  # self.populationsize,
                        self.deaths,
                        self.births,
                        sexs,
                        bt_ptCounts,
                        btrf_ptCounts,
                        ]

        self.states.append(currentstate)

    def log(self, strings):
        self.check_directory(self.plots_dir)
        with open(self.plots_dir+self.log_file, "a") as f:
            f.writelines(strings)

    def save_states(self):
        bt_pt = ['bt_pt-{}'.format(bt) for bt in self.bt_pt]
        btrf_pt = ['btrf_pt-{}'.format(btrf) for btrf in self.btrf_pt]

        cols = ['time',
                'population_size',
                'deaths',
                'births',
                'sex_f',
                'sex_m',
                ] + bt_pt + btrf_pt

        log_flattend = []
        for state in self.states:
            s_flatten = []
            for e in state:
                if type(e) is list:
                    s_flatten.extend(e)
                else:
                    s_flatten.append(e)
            log_flattend.append(s_flatten)

        log_df = pd.DataFrame(data=log_flattend, columns=cols)
        self.check_directory(self.plots_dir)
        log_df.to_csv(self.plots_dir+'log_df.csv', header=True, index=False)

    def check_directory(self, dir):
        if not os.path.isdir(dir):
            os.makedirs(dir)

    def updateSize(self):
        x = np.arange(0, self.time)
        y = [state[2] for state in self.states]
        self.ax[0].fill_between(x, y, np.zeros(len(y)))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_size(self, cumulativ=False, save=False):
        # x = np.arange(0, self.time, step=self.timestep)
        x = np.array([state[0] for state in self.states])
        y = np.array([state[1] for state in self.states])
        d = np.array([state[2] for state in self.states])
        b = np.array([state[3] for state in self.states])

        # import ipdb; ipdb.set_trace()
        fig, ax = plt.subplots()
        ax.fill_between(x,
                        y,
                        0,
                        color=default_colors[0],
                        alpha=1,
                        label='initial size' if cumulativ else 'populationsize')
        ax.fill_between(x,
                        y,
                        y - b,
                        color=default_colors[1],
                        alpha=1,
                        label='cumulative births' if cumulativ else 'total births')
        ax.fill_between(x,
                        y + d,
                        y,
                        color=default_colors[2],
                        alpha=1,
                        label='cumulative deaths' if cumulativ else 'total deaths')
        plt.legend()
        if save:
            self.check_directory(self.plots_dir)
            plt.savefig(self.plots_dir+"populationsize.png")
        else:
            plt.show()

    def plot_bt_pt(self, showRf=False, ratio=False, save=False):
        # x = np.cumsum([state[0] for state in self.states])
        x = np.array([state[0] for state in self.states])
        if showRf:
            y = np.cumsum([state[6] for state in self.states], axis=1)
        else:
            y = np.cumsum([state[5] for state in self.states], axis=1)
        if ratio:
            y = (y.T / y[:, -1]).T

        fig, ax = plt.subplots()
        if showRf:
            for i, p in enumerate(self.btrf_pt):
                # ax.plot(X, Y[:, i], color=self.bt_pt_colors[p], alpha=1.00, label=p, linewidth=0.1)
                ax.fill_between(x,
                                y[:, i],
                                np.zeros(
                                    y[:, 0].shape) if i == 0 else y[:, i - 1],
                                color=self.btrf_pt_colors[p],
                                alpha=1,
                                label=p
                                )
        else:
            for i, p in enumerate(self.bt_pt):
                # ax.plot(X, Y[:, i], color=self.bt_pt_colors[p], alpha=1.00, label=p, linewidth=0.1)
                ax.fill_between(x,
                                y[:, i],
                                np.zeros(
                                    y[:, 0].shape) if i == 0 else y[:, i - 1],
                                color=self.bt_pt_colors[p],
                                alpha=1,
                                label=p
                                )
        plt.legend()
        if save:
            self.check_directory(self.plots_dir)
            plt.savefig(self.plots_dir
                        + "bloodtype{}{}.png".format('_rf' if showRf else '',
                                                     '_ratio' if ratio else ''))
        else:
            plt.show()

    def plot_sex(self, ratio=False, save=False):
        # x = np.cumsum([state[0] for state in self.states])
        x = np.array([state[0] for state in self.states])
        y = np.cumsum([state[4] for state in self.states], axis=1)
        if ratio:
            y = (y.T / y[:, -1]).T

        fig, ax = plt.subplots()
        ax.fill_between(x, y[:, 0], np.zeros(y[:, 0].shape),
                        color=default_colors[1], alpha=1, label='f')
        ax.fill_between(x, y[:, 1], y[:, 0],
                        color=default_colors[0], alpha=1, label='m')
        plt.legend()
        if save:
            self.check_directory(self.plots_dir)
            plt.savefig(self.plots_dir
                        + "sex_distribution{}.png".format('_ratio' if ratio else ''))
        else:
            plt.show()

    def print_state(self):
        print("step: {:3.2f}".format(self.states[-1][0]),
              self.states[-1][1:]
              )
