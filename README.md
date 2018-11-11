# scenred
## Scenario reduction algorithms for stochastic programming.

Python implementation of the simultaneous backward reduction and fast forward selection scenario reduction techniques for stochastic programming. The algorithms itself are implemented in optimized cython code, which translates to C-code, for fast execution.
The algorithms itself are described in [1].

## Usage
One can use the scenario reduction starting from a 2d-numpy array, with the number of rows the length of each scenario and each column being one scenario.

```
import numpy as np
import scenario_reduction as scen_red
scenarios = np.random.rand(10,30)  # Create 30 random scenarios of length 10. 
probabilities = np.random.rand(30)
probabilities = probabilities/np.sum(probabilities)  # Create random probabilities of each scenario and normalize 

S = scen_red.ScenarioReduction(scenarios, probabilities=probabilities, cost_func='general', r = 2, scen0 = np.zeros(10))
S.fast_forward_sel(n_sc_red=5, num_threads = 4)  # use fast forward selection algorithm to reduce to 5 scenarios with 4 threads 
scenarios_reduced = S.scenarios_reduced  # get reduced scenarios
probabilities_reduced = S.probabilities_reduced  # get reduced probabilities

```


# References
[1]: [Heitsch, Holger, and Werner Römisch. "Scenario reduction algorithms in stochastic programming." Computational optimization and applications 24.2-3 (2003): 187-206.](https://edoc.hu-berlin.de/bitstream/handle/18452/3285/8.pdf?sequence=1)

