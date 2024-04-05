import numpy as np
import matplotlib.pyplot as plt
import configparser

config = configparser.ConfigParser()
config.read("./config.ini")

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif (\rmfamily)"

fontsize = 22
labelsize = fontsize
linewidth = 3

x = np.arange(0, 60 + 1, 10)
# selection = np.arange(0, x.shape[0], 3)
# selection = np.arange(0, x.shape[0], 1)

fig, ax = plt.subplots()

loss_val = np.load("./.results/test_loss_kalman_noise_hetero.npy")
print(loss_val.shape)
ax.plot(
    x,
    loss_val,
    label="KalmanNet",
    linewidth=linewidth,
    color="#0000ff",
    marker=">",
    markerfacecolor="none",
    markersize=10,
    linestyle="dashdot",
)

loss_val = np.load("./.results/test_loss_split_noise_hetero.npy")
print(loss_val.shape)
ax.plot(
    x,
    loss_val,
    label="Split-KalmanNet",
    linewidth=linewidth,
    color="#ff0000",
    marker="D",
    markerfacecolor="none",
    markersize=10,
    linestyle="solid",
)

loss_val = np.load("./.results/test_loss_ekf_noise_hetero.npy")
print(loss_val.shape)
ax.plot(
    x,
    loss_val,
    label="EKF (perfect)",
    linewidth=linewidth,
    color="#000000",
    marker="X",
    markerfacecolor="none",
    markersize=10,
    linestyle="dashed",
)

loss_val = np.load("./.results/test_loss_ekf_mm_noise_hetero.npy")
print(loss_val.shape)
ax.plot(
    x,
    loss_val,
    label="EKF (mismatch)",
    linewidth=linewidth,
    color="#abab00",
    marker="o",
    markerfacecolor="none",
    markersize=10,
    linestyle="dashed",
)

ax.xaxis.set_tick_params(labelsize=labelsize)
ax.yaxis.set_tick_params(labelsize=labelsize)
ax.set_xlabel(r"$\nu$ [dB]", fontsize=fontsize, fontweight="bold")
ax.set_ylabel("MSE [dB]", fontsize=fontsize, fontweight="bold")

ax.legend(fontsize=fontsize)
ax.grid()
plt.tight_layout()
# fig.savefig("./.results/convergence.eps", format="eps")

plt.show()
