#!/usr/bin/env python
"""
MCMC parameter analysis and correlation visualization for Er:YSO.

This example demonstrates post-processing of MCMC optimization results:
1. Load parameter and chi-squared samples from MCMC fitting
2. Apply burn-in period (discard initial steps)
3. Thin samples to remove correlation
4. Visualize parameter distributions and correlations
5. Organize parameters by crystal field tensors (r2, r4, r6)

Input files expected:
- 2018-08-08_eryso_site1_chi2accept.npy: chi-squared values
- 2018-08-08_eryso_site1_xaccept.npy: parameter samples

Prerequisites:
- MCMC fitting results (*.npy files)
- numpy, matplotlib
- Results from prior MCMC fitting run (e.g., mhfit_siman.py)
"""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

chi2 = np.load("2018-08-08_eryso_site1_chi2accept.npy")
x = np.load("2018-08-08_eryso_site1_xaccept.npy")

x = x[50000:]  # Discard initial 50000 steps, let the algorithm "burn in".
x = x[::1000]  # Use only every 1000 step, to ensure they are not corellated.

param_ord = [
    "EAVG",
    "ZETA",
    "F2",
    "F4",
    "F6",
    "C20",
    "C21",
    "C21I",
    "C22",
    "C22I",
    "C40",
    "C41",
    "C41I",
    "C42",
    "C42I",
    "C43",
    "C43I",
    "C44",
    "C44I",
    "C60",
    "C61",
    "C61I",
    "C62",
    "C62I",
    "C63",
    "C63I",
    "C64",
    "C64I",
    "C65",
    "C65I",
    "C66",
    "C66I",
    "HYP",
    "EQHYP",
]

cf = [
    ["C20"],
    ["C21", "C21I"],
    ["C22", "C22I"],
    ["C40"],
    ["C41", "C41I"],
    ["C42", "C42I"],
    ["C43", "C43I"],
    ["C44", "C44I"],
    ["C60"],
    ["C61", "C61I"],
    ["C62", "C62I"],
    ["C63", "C63I"],
    ["C64", "C64I"],
    ["C65", "C65I"],
    ["C66", "C66I"],
]

cf_r2 = [["C20"], ["C21", "C21I"], ["C22", "C22I"]]
cf_r4 = [["C40"], ["C41", "C41I"], ["C42", "C42I"], ["C43", "C43I"], ["C44", "C44I"]]
cf_r6 = [
    ["C60"],
    ["C61", "C61I"],
    ["C62", "C62I"],
    ["C63", "C63I"],
    ["C64", "C64I"],
    ["C65", "C65I"],
    ["C66", "C66I"],
]
fi = [["EAVG"], ["ZETA"], ["F2"], ["F4"], ["F6"], ["HYP"], ["EQHYP"]]

all_param = [
    ["EAVG"],
    ["ZETA"],
    ["F2"],
    ["F4"],
    ["F6"],
    ["C20"],
    ["C21", "C21I"],
    ["C22", "C22I"],
    ["C40"],
    ["C41", "C41I"],
    ["C42", "C42I"],
    ["C43", "C43I"],
    ["C44", "C44I"],
    ["C60"],
    ["C61", "C61I"],
    ["C62", "C62I"],
    ["C63", "C63I"],
    ["C64", "C64I"],
    ["C65", "C65I"],
    ["C66", "C66I"],
    ["HYP"],
    ["EQHYP"],
]

#### Select for which parameters you want to produce plots (from above lists)
plt_param = all_param

n = len(plt_param)
# Correlation between parameters
fig = plt.figure(1)
for i, pi in enumerate(plt_param):
    for j, pj in enumerate(plt_param):
        ax = fig.add_subplot(n, n, j + i * n + 1)
        for ii, pii in enumerate(pi):
            for jj, pij in enumerate(pj):
                ax.scatter(
                    x[:, param_ord.index(pii)],
                    x[:, param_ord.index(pij)],
                    label="l",
                    color="C%i" % (ii * 2 + jj),
                )
        if j == 0:
            ax.set_ylabel(pi[0])
        if i == n - 1:
            ax.set_xlabel(pj[0])

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, ["Re(x)-Re(y)", "Re(y)-Im(x)", "Re(x)-Im(y)", "Im(x)-Im(y)"], loc="upper right")


# Histograms of posterior distribution samples
fig = plt.figure(2)

for i, pi in enumerate(plt_param):
    for ii, pii in enumerate(pi):
        ax = fig.add_subplot(n, 2, i * 2 + ii + 1)
        ax.hist(x[:, param_ord.index(pii)])
        s = np.std(x[:, param_ord.index(pii)])
        mu = np.mean(x[:, param_ord.index(pii)])
        ax.text(
            0.75,
            0.8,
            r"$\mu=%.2f,\ \sigma=%.2f$" % (mu, s),
            transform=ax.transAxes,
            fontweight="bold",
        )
        ax.set_yticklabels([])
        if ii == 0:
            ax.set_ylabel(pi[0])
        if i == n - 1:
            if ii == 0:
                ax.set_xlabel("Re(p)")
            else:
                ax.set_xlabel("Im(p)")

uncert = []
for i, pi in enumerate(plt_param):
    if len(pi) == 2:
        s = np.std(x[:, param_ord.index(pi[0])]) + np.std(x[:, param_ord.index(pi[1])]) * complex(
            0, 1
        )
    else:
        s = np.std(x[:, param_ord.index(pi[0])])
    uncert += [s]

print("Parameter uncertainties:")
print(uncert)


## Autocorrelation of x vectors
def acf(x):
    return np.array([1] + [np.corrcoef(x[:-i], x[i:])[0, 1] for i in range(1, len(x))])


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for i in range(x.shape[1]):
    ax.plot(acf(x[:, i]))


## Plot change of chi2 wrt accepted iteration number
fig = plt.figure(4)
ax = fig.add_subplot(1, 1, 1)
ax.plot(chi2)
plt.tight_layout()


plt.show()
