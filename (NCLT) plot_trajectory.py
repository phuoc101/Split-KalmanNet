import torch
import matplotlib.pyplot as plt
import os
import numpy as np

if not os.path.exists("./.results"):
    os.mkdir("./.results")


def rmse(pred, label, ax):
    return np.round(np.sqrt(((pred - label) ** 2).mean(axis=ax)), 3)


plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif (\rmfamily)"

n = 0

fig, ax = plt.subplots()

TEST_PATH = "./.data/NCLT/test"

data_x = torch.load(os.path.join(TEST_PATH, "state.pt"))
print(data_x.shape)
x_true = data_x[n][0][1:].numpy()
y_true = data_x[n][3][1:].numpy()
xy_true = np.vstack([x_true, y_true])
ax.scatter(x_true[0], y_true[0], label="Start", c="#80ff80", marker="*", s=300)
ax.scatter(x_true[-1], y_true[-1], label="End", c="#ff8080", marker="*", s=300)
ax.plot(
    x_true,
    y_true,
    label="Ground truth",
    linewidth=2,
    color="#808080",
    linestyle="solid",
)


data_x = torch.load(os.path.join(TEST_PATH, "KF v1 x_hat.pt"))
x_kn = data_x[n][0][1:].numpy()
y_kn = data_x[n][3][1:].numpy()
xy_kn = np.vstack([x_kn, y_kn])
ax.plot(
    x_kn,
    y_kn,
    label=f"KalmanNet - RMSE {rmse(xy_kn, xy_true, 1)} m",
    linewidth=2,
    color="#0000ff",
    linestyle="dashdot",
)


data_x = torch.load(os.path.join(TEST_PATH, "SKF x_hat.pt"))
x_skn = data_x[n][0][1:].numpy()
y_skn = data_x[n][3][1:].numpy()
xy_skn = np.vstack([x_skn, y_skn])
ax.plot(
    x_skn,
    y_skn,
    label=f"Split-KalmanNet - RMSE {rmse(xy_skn, xy_true, 1)} m",
    linewidth=2,
    color="#ff0000",
    linestyle="solid",
)

data_x = torch.load(os.path.join(TEST_PATH, "EKF x_hat.pt"))
x_ekf = data_x[n][0][1:].numpy()
y_ekf = data_x[n][3][1:].numpy()
xy_ekf = np.vstack([x_ekf, y_ekf])
ax.plot(
    x_ekf,
    y_ekf,
    label=f"EKF (mismatch) - RMSE {rmse(xy_ekf, xy_true, 1)} m",
    linewidth=2,
    color="#00ff00",
    linestyle="dotted",
)


# data_x = torch.load('./.data/NCLT/test/KF v2 x_hat.pt')
# x = data_x[n][0][1:].numpy()
# y = data_x[n][3][1:].numpy()
# ax.plot(x, y, label='KalmanNet 2')


ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
ax.set_xlabel("x [m]", fontsize=14, fontweight="bold")
ax.set_ylabel("y [m]", fontsize=14, fontweight="bold")

ax.legend(fontsize=14)
ax.grid()

plt.tight_layout()

# fig.savefig("./.results/nclt_trajectory.svg", format="svg")

plt.show()
