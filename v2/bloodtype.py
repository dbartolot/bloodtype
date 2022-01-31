
import numpy as np
from matplotlib import pyplot as plt
import random
import os
import numba as nb
import math
import pandas as pd
from tqdm import trange
import time
import sys


default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf', '#1a55FF']

df_cols = ['age',
           'sex',
           'bt_gt',
           'bt_pt',
           'rf_gt',
           'rf_pt',
           'weights',
           ]


SCALER = 1.4  # birth/death scaler prevents stagnation in evolution
DEATH_RATE = 0.0094 * SCALER
BIRTH_RATE = 0.0096 * SCALER

btrf_austria = {
    "A+":  0.37,
    "B+":  0.12,
    "O-":  0.06,
    "O+":  0.30,
    "A-":  0.07,
    "B-":  0.02,
    "AB+": 0.05,
    "AB-": 0.01,
}


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
    death_rate = DEATH_RATE
    deaths = 0
    birth_rate = BIRTH_RATE
    births = 0
    weights = pd.Series({'O': 1, 'A': 1, 'B': 1, 'AB': 1})
    plots_dir = './plots/'
    filename_age_penalty = 'age_penalty.csv'
    filename_birth_distribution = 'birth_distribution.csv'
    log_file = "log.txt"
    min_off_score = [sys.maxsize, 0.0, []]

    def __init__(self,
                 startsize,
                 timesteptype='y',
                 death_rate=DEATH_RATE,
                 birth_rate=BIRTH_RATE,
                 age_based_model=True,
                 ):
        """Initialize the model.

        Parameters
        ----------
        startsize : int
            Size of the starting population.
        timesteptype : char
            Indicates the simulation steps, which can be 'd', 'w', 'm' and 'y'
            for days, weeks, months and years respectively.
            Default is 'y'.
        death_rate : float
            The `death_rate` expresses the number of people which die per
            person. Calculated by the number of deaths in a country divided by
            their average population size. Should usually be between 0 and 1.
            Default is `DEATH_RATE`.
        birth_rate : float
            The `birth_rate` expresses the number of people which are born per
            person. Calculated by the number of births in a country divided by
            their average population size. Should usually be between 0 and 1.
            Default is `BRITH_RATE`.
        age_based_model : bool
            If True the age of every person is taken into account by selecting
            who gets to die and who gives birth.
            Default is True.

        """
        self.states = []
        self.population = []

        self.timestep = 7 / 365
        self.timesteptype = 'w'
        self.time = 0
        self.deaths = 0
        self.births = 0

        self.age_based_model = age_based_model

        self.check_directory(self.plots_dir)
        t = time.localtime()
        self.plots_dir = './plots/' + time.strftime('%Y-%b-%d_%H%M', t) + '/'
        self.check_directory(self.plots_dir)

        self.set_timesteps(timesteptype)
        self.load_age_penalty()
        self.load_birth_distribution()

        self.set_death_rate(death_rate)
        self.set_birth_rate(birth_rate)
        # plt.ion()
        # self.fig, self.ax = plt.subplots(nrows=1, ncols=2)

        self.populationsize = startsize
        self.population = self.gen_population(age='rand',
                                              size=self.populationsize)

    def gen_population(self,
                       age='rand',
                       sex=None,
                       bt_gt='OO',
                       rf_gt='++',
                       offspring_prob=None,
                       size=1
                       ):

        if type(age) is float and age >= 0:
            data = [self.gen_person(age, sex, bt_gt, rf_gt)
                    for i in range(size)]
        else:
            data = [self.gen_person(random.random() * 50, sex, bt_gt, rf_gt)
                    for i in range(size)]
        df = pd.DataFrame(data=data,
                          columns=df_cols)
        return df

    def gen_person(self,
                   age=0.0,
                   sex=None,
                   bt_gt='OO',
                   rf_gt='++',
                   ):

        if sex not in ['f', 'm']:
            sex = 'f' if random.random() < 0.5 else 'm'

        bt_pt = self.get_bt_pt(bt_gt)
        rf_pt = self.get_rf_pt(rf_gt)

        weights_factor = self.get_weights(bt_pt)

        ret = [age, sex, bt_gt, bt_pt, rf_gt,
               rf_pt, weights_factor]
        return ret

    def get_weights(self, bt_pt):
        return self.weights[bt_pt]

    def log(self, string):
        self.check_directory(self.plots_dir)
        with open(self.plots_dir + self.log_file, "a") as f:
            f.writelines(string)

    @staticmethod
    def get_bt_pt(bt_gt):
        if bt_gt == 'OO':
            return 'O'
        elif bt_gt == 'AA' or bt_gt == 'AO' or bt_gt == 'OA':
            return 'A'
        elif bt_gt == 'BB' or bt_gt == 'BO' or bt_gt == 'OB':
            return 'B'
        elif bt_gt == 'AB' or bt_gt == 'BA':
            return 'AB'
        else:
            return None

    @staticmethod
    def get_rf_pt(rf_gt):
        if rf_gt == '++' or rf_gt == '-+' or rf_gt == '+-':
            return '+'
        else:
            return '-'

    def set_weights(self, weights):
        self.weights = pd.Series(weights)
        self.log("#{}#: Weights = {}\n".format(self.time, self.weights))

    def set_timesteps(self, timesteptype):
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

    def load_age_penalty(self):
        self.age_penalty = np.genfromtxt(self.filename_age_penalty,
                                         delimiter=',',
                                         skip_header=1)
        prop = (self.age_penalty[:, 2]
                / np.sum(self.age_penalty[:, 2])).reshape(-1, 1)
        self.age_penalty = np.hstack([self.age_penalty, prop])

    def load_birth_distribution(self):
        self.birth_distribution = np.genfromtxt(self.filename_birth_distribution,
                                                delimiter=',',
                                                skip_header=1)
        prop = (self.birth_distribution[:, 2]
                / np.sum(self.birth_distribution[:, 2])).reshape(-1, 1)
        cdf = np.cumsum(prop).reshape(-1, 1)
        self.birth_distribution = np.hstack([self.birth_distribution,
                                             prop,
                                             cdf])

    def step(self, bt_mutation=None, rf_mutation=None, mutations=None):
        # population size till greater than 0?
        if self.populationsize < 1:
            return

        # update time
        self.time += self.timestep

        # update age of population
        self.population['age'] += self.timestep

        # deaths
        self.kill_persons()

        # population size till greater than 0?
        if self.populationsize < 1:
            # self.log_state()
            del self.states[-1]
            return

        # births
        self.generate_offsprings(bt_mutation=bt_mutation,
                                 rf_mutation=rf_mutation,
                                 mutations=mutations)

        # log state
        self.log_state()

    def steps(self, steps, bt_mutation=None, rf_mutation=None,
              mutations=None, print_state=True, show_off_score=True):
        """Simulate `steps` amount of time steps.

        Parameters
        ----------
        steps : int
            Number of individual time steps executed by the simulation.
        bt_mutation : char
            Gives every offspring during the steps a mutated blood type of the
            `bt_mutation`, which is either 'A' or 'B'.
            If `bt_mutation` is not set or set to None no mutations for the
            blood type are introduced.
        rf_mutation : char
            Gives every offspring during the steps a mutated blood type of the
            `rf_mutation`, which is either '+' or '-'.
            If `rf_mutation` is not set or set to None no mutations for the
            blood type are introduced.
        mutations : int
            The number of offsprings which shoudl carry the bt or rf mutation.
            If the value is None or greater than the number of births during
            an iteration, the value gets set to the births amount of mutations.
        print_state : bool
            Prints the state of the last step after finischen all time steps.
        show_off_score : bool
            Shows always a score which indicateds by how much the current
            distribution of blood types and rhesus factors is off from the
            austrian distribution.
        """
        t = trange(steps)
        off_score = self.min_off_score[0]
        for i in t:
            if show_off_score:
                t.set_description("Population Size {}".format(self.populationsize)
                                  + ", year {:.1f}".format(self.time)
                                  + ", off_score {:.2f}".format(off_score))
            else:
                t.set_description("Population Size {}".format(self.populationsize)
                                  + ", year {:.1f}".format(self.time))
            t.refresh()
            self.step(bt_mutation=bt_mutation,
                      rf_mutation=rf_mutation,
                      mutations=mutations)

            btrf_dist = self.states[-1][6] / sum(self.states[-1][6])
            off_score = sum([abs(btrf_dist[i] / btrf_austria[btrf] - 1)
                             for i, btrf in enumerate(self.btrf_pt)])
            if self.min_off_score[0] > off_score:
                self.min_off_score = [off_score, self.time, btrf_dist]

        if show_off_score:
            print("year {:.1f}".format(self.min_off_score[1])
                  + ", off_score {:.2f}".format(self.min_off_score[0])
                  + ", btrf distribution {}".format(self.min_off_score[2])
                  )
        if print_state:
            self.print_state()

    def get_deathlist(self):
        prop_bt = self.weights[self.population['bt_pt']]
        prop_bt = prop_bt / np.sum(prop_bt)

        if self.age_based_model:
            ap = self.age_penalty
            ages = self.population['age']

            prop_age = np.array([ap[((ap[:, 0] - age) <= 0).sum() - 1, 3]
                                 for age in ages])
            prop_age = prop_age / np.sum(prop_age)
            prop = np.array(prop_bt * prop_age)
        else:
            prop = prop_bt

        prop = prop / np.sum(prop)

        self.deaths = self.populationsize * self.get_death_rate()
        if random.random() < self.deaths - int(self.deaths):
            self.deaths = math.ceil(self.deaths)
        else:
            self.deaths = round(self.deaths)

        def indexlist(deaths, prop):
            indexList = np.empty((deaths), dtype=np.int32)
            for i in nb.prange(deaths):
                # for i in range(deaths):
                cdf = np.cumsum(prop)
                cdf = cdf / cdf[-1]
                indexList[i] = np.sum(cdf - random.random() < 0)
                prop[indexList[i]] = 0
            return np.flip(np.sort(indexList))
        return self.population.index[indexlist(self.deaths, prop)]

    def kill_persons(self):
        deathlist = self.get_deathlist()
        if deathlist.shape != (0,):
            for i in deathlist:
                self.remove_person(i)

    def remove_person(self, index):
        self.population = self.population.drop(index)
        self.populationsize = self.populationsize - 1

    def add_person(self, person):
        self.population = pd.concat([self.population, person],
                                    ignore_index=True)
        self.populationsize = self.populationsize + 1

    def add_persons(self, new_persons):
        self.population = pd.concat([self.population, new_persons],
                                    ignore_index=True)
        self.populationsize = self.populationsize + new_persons.shape[0]

    def generate_offsprings(self,
                            bt_mutation=None,
                            rf_mutation=None,
                            mutations=0):
        self.births = self.populationsize * self.get_birth_rate()
        if random.random() < self.births - int(self.births):
            self.births = math.ceil(self.births)
        else:
            self.births = round(self.births)

        if bt_mutation not in self.bt_pt and \
                rf_mutation not in self.rf_pt:
            mutations = 0

        # reduce number of mutations to the proper size of births
        if (mutations is None) or (self.births < mutations):
            mutations = self.births

        parents = self.choose_parents(self.births)

        children = [None] * self.births

        for i in range(mutations):
            pid1, pid2 = parents[i]
            if (pid1, pid2) == (-1, -1):
                break
            children[i] = self.gen_mutated_offspring(
                pid1, pid2, bt_mutation, rf_mutation)

        for i in range(mutations, self.births):
            pid1, pid2 = parents[i]
            if (pid1, pid2) == (-1, -1):
                break
            children[i] = self.gen_offspring(pid1, pid2)

        children = [child for child in children if child is not None]
        self.births = len(children)

        children = pd.DataFrame(data=children, columns=df_cols)
        self.add_persons(children)

    def gen_offspring(self, pid1, pid2):
        child_bt_gt = self.population['bt_gt'][pid1][random.getrandbits(1)] + \
            self.population['bt_gt'][pid2][random.getrandbits(1)]
        child_rf_gt = self.population['rf_gt'][pid1][random.getrandbits(1)] + \
            self.population['rf_gt'][pid2][random.getrandbits(1)]

        child = self.gen_person(age=0.0,
                                sex=None,
                                bt_gt=child_bt_gt,
                                rf_gt=child_rf_gt,
                                )
        return child

    def gen_mutated_offspring(self,
                              pid1,
                              pid2,
                              bt_gtMutation=None,
                              rf_gtMutation=None):

        child = self.gen_offspring(pid1, pid2)

        if bt_gtMutation is not None:
            # bt_gt is position 2
            child[2] = self.mutate_bt_gt(child[2], bt_gtMutation)
            # bt_pt is position 3
            child[3] = self.get_bt_pt(child[2])
        if rf_gtMutation is not None:
            # rf_pt is position 4
            child[4] = self.mutate_rf_gt(child[4], rf_gtMutation)
            # rf_pt is position 5
            child[5] = self.get_rf_pt(child[4])

        return child

    @staticmethod
    def mutate_bt_gt(bt_gt, mutateTo):
        return BloodType.mutate_gt(bt_gt, mutateTo)

    @staticmethod
    def mutate_rf_gt(rf_gt, mutateTo):
        return BloodType.mutate_gt(rf_gt, mutateTo)

    @staticmethod
    def mutate_gt(gt, mutateTo):
        if bool(random.getrandbits(1)):
            return gt[0] + mutateTo
        else:
            return mutateTo + gt[1]

    def choose_parents(self, size=1):
        sexs = np.array(self.population['sex'])

        if self.age_based_model:
            bd = self.birth_distribution
            ages = self.population['age']
            prop_i = np.array([bd[((bd[:, 0] - age) <= 0).sum() - 1, 3]
                               for age in ages])
        else:
            prop_i = np.ones(self.population.shape[0])

        prop_i = prop_i / np.sum(prop_i)

        def indexlist(sexs, prop_i, size=1):
            ij = np.empty((size, 2), dtype=np.int64)
            cdf_i = np.cumsum(prop_i)
            cdf_i = cdf_i / cdf_i[-1]
            for i in nb.prange(size):
                ij[i, 0] = np.sum(cdf_i - random.random() < 0)

                prop_j = prop_i * (sexs != sexs[i])
                cdf_j = np.cumsum(prop_j)
                cdf_j = cdf_j / cdf_j[-1]
                ij[i, 1] = np.sum(cdf_j - random.random() < 0)
            return ij

        index_list = indexlist(sexs, prop_i, size)
        parents_ij = [[self.population.index[i], self.population.index[j]]
                      for i, j in index_list]

        return parents_ij

    def log_state(self):
        bt_ptFromPopulation = self.population['bt_pt']
        bt_ptCounts = [bt_ptFromPopulation.str.count(p).sum()
                       for p in self.bt_pt]

        btrf_ptFromPopulation = self.population['bt_pt'] + \
            self.population['rf_pt']
        btrf_ptCounts = [btrf_ptFromPopulation.str.count(p).sum()
                         for p in self.btrf_pt]
        sexs = [np.sum(self.population['sex'] == 'f').sum(), 0]
        sexs[1] = self.populationsize - sexs[0]

        age_group_bins = np.append(self.age_penalty[:, 0],
                                   np.max([self.population['age'].max(),
                                          self.age_penalty[-1, 0] + 10]))
        age_groups = np.histogram(self.population['age'],
                                  bins=age_group_bins)[0].tolist()

        currentstate = [self.time,
                        self.populationsize,
                        self.deaths,
                        self.births,
                        sexs,
                        bt_ptCounts,
                        btrf_ptCounts,
                        age_groups,
                        ]

        self.states.append(currentstate)

    def save_states(self):
        age_groups = ['[{}{})'.format(ag[0], '' if i == self.age_penalty.shape[0] - 1 else '-{}'.format(ag[1]))
                      for i, ag in enumerate(self.age_penalty)]
        bt_pt = ['bt_pt-{}'.format(bt) for bt in self.bt_pt]
        btrf_pt = ['btrf_pt-{}'.format(btrf) for btrf in self.btrf_pt]

        cols = ['time',
                'population_size',
                'deaths',
                'births',
                'sex_f',
                'sex_m',
                ] + bt_pt + btrf_pt + age_groups

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
        log_df.to_csv(self.plots_dir + 'log_df.csv', header=True, index=False)

    @staticmethod
    def check_directory(dir):
        if not os.path.isdir(dir):
            os.makedirs(dir)

    def updateSize(self):
        x = np.arange(0, self.time)
        y = [state[2] for state in self.states]
        self.ax[0].fill_between(x, y, np.zeros(len(y)))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_size(self, cumulativ=False, save=False):
        x = np.array([state[0] for state in self.states])
        y = np.array([state[1] for state in self.states])
        d = np.array([state[2] for state in self.states])
        b = np.array([state[3] for state in self.states])

        plt.rc('font', size=14)
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.subplots_adjust(right=0.6)

        twin1 = ax.twinx()
        twin2 = ax.twinx()

        # Offset the right spine of twin2.  The ticks and label have already
        # been placed on the right by twinx above.
        twin2.spines.right.set_position(("axes", 1.25))

        p1, = ax.plot(x, y, default_colors[0], label="population size")
        p2, = twin1.plot(x, b, default_colors[1], label="births")
        p3, = twin2.plot(x, d, default_colors[2], label="deaths")

        ax.set_xlabel("year")
        ax.set_ylabel("population size")
        twin1.set_ylabel("births")
        twin2.set_ylabel("deaths")

        ax.yaxis.label.set_color(p1.get_color())
        twin1.yaxis.label.set_color(p2.get_color())
        twin2.yaxis.label.set_color(p3.get_color())

        tkw = dict(size=4, width=0.5)
        ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
        twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
        twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
        ax.tick_params(axis='x', **tkw)

        plt.xlabel("years")
        plt.tight_layout()

        if save:
            fig.set_dpi(400.0)
            plt.savefig(self.plots_dir + "populationsize.png")
            plt.close()
        else:
            plt.show()

    def plot_bt_pt(self, showRf=False, ratio=False, save=False):
        x = np.array([state[0] for state in self.states])

        if showRf:
            y = np.array([state[6] for state in self.states])
        else:
            y = np.array([state[5] for state in self.states])
        if ratio:
            y = np.cumsum(y, axis=1)
            y = (y.T / y[:, -1]).T

        plt.rc('font', size=14)
        fig, ax = plt.subplots(figsize=(7, 5))
        if ratio:
            if showRf:
                for i in reversed(range(len(self.btrf_pt))):
                    p = self.btrf_pt[i]
                    y_upper = y[:, i]
                    y_lower = np.zeros(
                        y[:, 0].shape) if i == 0 else y[:, i - 1]

                    ax.fill_between(x,
                                    y_upper,
                                    y_lower,
                                    color=self.btrf_pt_colors[p],
                                    alpha=1,
                                    label=p
                                    )
                plt.ylabel("ratio of population size per blood type \n"
                           + "and rhesus factor")
            else:
                for i in reversed(range(len(self.bt_pt))):
                    p = self.bt_pt[i]
                    y_upper = y[:, i]
                    y_lower = np.zeros(
                        y[:, 0].shape) if i == 0 else y[:, i - 1]
                    ax.fill_between(x,
                                    y_upper,
                                    y_lower,
                                    color=self.bt_pt_colors[p],
                                    alpha=1,
                                    label=p
                                    )
                plt.ylabel("ratio of population size per blood type \n"
                           + "and rhesus factor")
        else:
            if showRf:
                for i in reversed(range(len(self.btrf_pt))):
                    p = self.btrf_pt[i]
                    ax.plot(x, y[:, i], color=self.btrf_pt_colors[p],
                            label=p)
                plt.ylabel("population size per blood type and rhesus factor")
            else:
                for i in reversed(range(len(self.bt_pt))):
                    p = self.bt_pt[i]
                    ax.plot(x, y[:, i], color=self.bt_pt_colors[p],
                            label=p)
                plt.ylabel("population size per blood type")
        plt.legend()
        plt.xlabel("years")
        plt.tight_layout()
        if save:
            self.check_directory(self.plots_dir)
            fig.set_dpi(400.0)
            plt.savefig(self.plots_dir + "bloodtype"
                        + "{}{}.png".format('_rf' if showRf else '',
                                            '_ratio' if ratio else ''))
            plt.close()
        else:
            plt.show()

    def plot_sex(self, ratio=False, save=False):
        x = np.array([state[0] for state in self.states])
        y = np.array([state[4] for state in self.states])
        if ratio:
            y = np.cumsum(y, axis=1)
            y = (y.T / y[:, -1]).T

        plt.rc('font', size=14)
        fig, ax = plt.subplots(figsize=(7, 5))
        if ratio:
            ax.fill_between(x, y[:, 1], y[:, 0],
                            color=default_colors[0], alpha=1, label='m')
            ax.fill_between(x, y[:, 0], np.zeros(y[:, 0].shape),
                            color=default_colors[1], alpha=1, label='f')
            plt.ylabel("ratio of sexs")
        else:
            ax.plot(x, y[:, 1], color=default_colors[0], label='m')
            ax.plot(x, y[:, 0], color=default_colors[1], label='f')
            plt.grid(linestyle='--', alpha=0.7)
            plt.ylabel("sexs")

        plt.legend()
        plt.xlabel("years")
        plt.tight_layout()
        if save:
            self.check_directory(self.plots_dir)
            fig.set_dpi(400.0)
            plt.savefig(self.plots_dir + "sex_distribution"
                        + "{}.png".format('_ratio' if ratio else ''))
            plt.close()
        else:
            plt.show()

    def plot_age_groups(self, ratio=False, save=False):
        x = np.array([state[0] for state in self.states])
        y = np.array([state[7] for state in self.states])

        if ratio:
            y = np.cumsum(y, axis=1)
            y = (y.T / y[:, -1]).T

        plt.rc('font', size=14)
        fig, ax = plt.subplots(figsize=(7, 5))

        if ratio:
            for i in reversed(range(y.shape[1])):
                if i == y.shape[1] - 1:
                    label = '{}-{}'.format(int(self.age_penalty[i, 0]),
                                           r'$\infty$')
                else:
                    label = '{}-{}'.format(int(self.age_penalty[i, 0]),
                                           int(self.age_penalty[i, 1]))
                y_upper = y[:, i]
                y_lower = np.zeros(y[:, 0].shape) if i == 0 else y[:, i - 1]
                ax.fill_between(x,
                                y_upper,
                                y_lower,
                                color=default_colors[i],
                                alpha=1,
                                label=label,
                                )
            plt.ylabel("ratio of population size \nper age group")
        else:
            for i in reversed(range(y.shape[1])):
                if i == y.shape[1] - 1:
                    label = '{}-{}'.format(int(self.age_penalty[i, 0]),
                                           r'$\infty$')
                else:
                    label = '{}-{}'.format(int(self.age_penalty[i, 0]),
                                           int(self.age_penalty[i, 1]))
                ax.plot(x,
                        y[:, i],
                        color=default_colors[i],
                        alpha=1,
                        label=label,
                        )
            plt.ylabel("population size per age group")
            plt.grid(linestyle='--', alpha=0.7)

        plt.legend()
        plt.xlabel("years")
        plt.tight_layout()
        if save:
            self.check_directory(self.plots_dir)
            fig.set_dpi(400.0)
            plt.savefig(self.plots_dir + "age_distribution"
                        + "{}.png".format('_ratio' if ratio else ''))
            plt.close()
        else:
            plt.show()

    def print_state(self):
        print("step: {:3.2f}".format(self.states[-1][0]), self.states[-1][1:])
