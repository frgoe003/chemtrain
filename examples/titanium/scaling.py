import matplotlib.pyplot as plt

import numpy as onp



n_mpl = 4
r_cutoff = 0.5

max_inner = 25.
min_inner = 0.1

# Compute the size of the whole domain
boundary = (2 * n_mpl + 1) * r_cutoff


outer = onp.linspace(2 * boundary + min_inner, 2 * boundary + max_inner, 100)

plt.plot(outer, (outer - 2 * boundary) ** 3 / outer ** 3)
plt.hlines(0.125, 0, 2 * boundary + max_inner)

twin = plt.twinx()
twin.plot(outer, outer ** 3 / (outer - 2 * boundary) ** 3)
twin.set_ylim([0, 25])

# Results: To get a benefit of running on 8 devices, we would need at least
#          a total box size of ~20

plt.show()

