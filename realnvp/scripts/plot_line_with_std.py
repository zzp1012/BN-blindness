import os
import numpy as np
import matplotlib.pyplot as plt

RES_PATH_LN = "NEED TO BE FILLED"
RES_PATH_BN = "NEED TO BE FILLED"

# load the two means
mean_ln = np.load(os.path.join(RES_PATH_LN, "mean_log_ll.npy"))
mean_bn = np.load(os.path.join(RES_PATH_BN, "mean_log_ll.npy"))

# load the two std
std_ln = np.load(os.path.join(RES_PATH_LN, "std_log_ll.npy"))
std_bn = np.load(os.path.join(RES_PATH_BN, "std_log_ll.npy"))

# load the alphas
alphas_ln = np.load(os.path.join(RES_PATH_LN, "alphas.npy"))
alphas_bn = np.load(os.path.join(RES_PATH_BN, "alphas.npy"))
assert np.allclose(alphas_ln, alphas_bn), "The two alphas are not the same!"
alphas = alphas_ln

# the 1 sigma upper and lower analytic population bounds
lower_bound_ln = mean_ln - 1.0 * std_ln
upper_bound_ln = mean_ln + 1.0 * std_ln
lower_bound_bn = mean_bn - 1.0 * std_bn
upper_bound_bn = mean_bn + 1.0 * std_bn

fig, ax = plt.subplots(1)
ax.plot(alphas, mean_ln, lw=1, label='ln', color='blue')
ax.fill_between(alphas, lower_bound_ln, upper_bound_ln, alpha=0.2, color='blue')
ax.plot(alphas, mean_bn, lw=1, label='bn', color='red')
ax.fill_between(alphas, lower_bound_bn, upper_bound_bn, alpha=0.2, color='red')
ax.legend(loc='upper left')
ax.set_xlabel('alphas')
ax.set_ylabel('log_ll')
ax.grid()
# save the fig
path = os.path.join(".", "{}.png".format("interpolation"))
fig.savefig(path)
plt.close()