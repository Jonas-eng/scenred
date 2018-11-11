import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, inplace=True)
from scenario_reduction_c import *


class ScenarioReduction():
    def __init__(self, scenarios, probabilities=None, cost_func='2norm', r=1, scen0=None):
        '''
        Class to perform scenario reduction
        :param scenarios: 2-dimensional numpy matrix of size n x m with n the lenght of one scenario and m the number
        of scenarios.
        :param probabilities: The probabilities of the initial scenarios. If None it is assumed the probabilities of
        all scenarios are equal. (default: None).
        :param cost_func: the cost function to calculated the Kantorovich distance. (default: eucledian norm)
        :param r: exponent in calculating the general cost function. (default: 1)
        :param scen0: the scenario_0 around which to calculate the general cost function. If none it is assumed to be 0.
        (default: None).
        '''
        if not(scenarios.flags['F_CONTIGUOUS']):
            scenarios = np.asfortranarray(scenarios)
        self.scenarios = scenarios
        self.n_sc = self.scenarios.shape[-1]
        if probabilities is None:
            probabilities = 1 / self.n_sc * np.ones(self.n_sc)
        self.probabilities = probabilities
        self.n_sc_red = self.n_sc
        self.cost_func = cost_func
        self.r = r
        self.scen0 = scen0
        self.cost_matrix = None
        self.idx_keep = None
        self.idx_del = None
        self.probabilities_reduced = None
        self.Kantorovich_distance = None

    @property
    def scenarios_reduced(self):
        if self.idx_keep is not None:
            return self.scenarios[:, self.idx_keep]
        else:
            return self.scenarios

    def calc_cost_matrix(self, num_threads=1, cost_func=None, r=1, scen0=None, verbose=0):

        if cost_func is not None:
            self.cost_func = cost_func
            self.r = r
            self.scen0 = scen0

        c = cost_matrix_c_par(self.scenarios, num_threads, cost_func=self.cost_func, r=self.r, scen0=self.scen0,
                              verbose=verbose)
        c_np = np.array(c)
        c_np[np.arange(self.n_sc, dtype=int), np.arange(self.n_sc, dtype=int)] = np.nan

        self.cost_matrix = c_np

        return c_np

    def _scenario_red(self, algorithm, n_sc_red, num_threads=1, verbose=0):
        self.n_sc_red = n_sc_red

        if self.cost_matrix is None:
            self.calc_cost_matrix(num_threads=num_threads, verbose=verbose)

        idx_del, idx_keep = algorithm(self.scenarios, n_sc_red, self.cost_matrix, self.probabilities,
                                                      num_threads=num_threads, verbose=verbose)
        self.idx_del = idx_del
        self.idx_keep = idx_keep

        self.probabilities_reduced = self.redistribute(self.cost_matrix, self.probabilities, idx_keep, idx_del)

        self.Kantorovich_distance = self.calc_Kantorovich_dist(self.cost_matrix, self.probabilities, self.idx_del,
                                                               self.idx_keep)

        return self.scenarios_reduced, self.probabilities_reduced

    def simult_backward_red(self, n_sc_red, num_threads=1, verbose=0):
        # from 'Heitsch, Holger, and Werner Römisch. "Scenario reduction algorithms in stochastic programming."
        # Computational optimization and applications 24.2-3 (2003): 187-206.'
        # p.192, algorithm 2.2
        return self._scenario_red(simult_backward_red_c_par, n_sc_red, num_threads, verbose)

    def fast_forward_sel(self, n_sc_red, num_threads=1, verbose=0):
        return self._scenario_red(fast_forward_sel_c_par, n_sc_red, num_threads, verbose)

    @staticmethod
    def calc_Kantorovich_dist(cost_matrix, probs, idx_del, idx_keep):
        Kantorovich_dist = np.sum(probs[idx_del]*np.min(cost_matrix[idx_del, :][:, idx_keep], 1))
        return Kantorovich_dist

    @staticmethod
    def redistribute(cost_matrix, probs, idx_keep, idx_del):
        # redistribution of probabilities
        probs_red = probs.copy()
        idx_del_closest = np.array(idx_keep)[np.nanargmin(cost_matrix[idx_del, :][:, idx_keep], 1)]
        for ix in range(len(idx_del)):
            probs_red[idx_del_closest[ix]] += probs_red[idx_del[ix]]
        probs_red = probs_red[idx_keep]

        return probs_red
