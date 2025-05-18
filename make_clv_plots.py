import numpy as np
import matplotlib.pyplot as plt

clvs = np.load('chaotic_soliton_clvs.npy')[70:-70,70:-70,:]
glider = np.load('chaotic_soliton_final.npy')[70:-70,70:-70]
LEs = np.loadtxt('chaotic_soliton_les.txt')
fig, axes = plt.subplots(3, 5, figsize=(15, 9))

def setup_empty_axis(ax):
    ax.axis('off')

fig, ax6 = plt.subplots(3, 6, figsize=(15, 9), gridspec_kw={'width_ratios':[1, 0.28, 1, 1, 1, 1], 'wspace':0})
for r in range(3):
    ax6[r, 1].axis('off')
axes = np.delete(ax6, 1, axis=1)

def plot_clv(ax, data, index):
    im = ax.imshow(data[..., index], cmap='bwr', vmin=-np.max(np.abs(data[..., index])), vmax=np.max(np.abs(data[..., index])))
    ax.set_xticks([])
    ax.set_yticks([])
    title = f"$\\lambda_{{{index+1}}}\\approx{LEs[index]:.4f}$" if abs(LEs[index]) >= 1e-7 else f"$\\lambda_{{{index+1}}}\\approx 0$"
    ax.set_title(title, fontsize=18)
    return im

def plot_glider(ax, data):
    ax.set_title("$A(t)$", fontsize=22)
    im = ax.imshow(data, cmap='gray_r')
    ax.set_xticks([])
    ax.set_yticks([])
    return im

# Row 1
setup_empty_axis(axes[0, 0])
setup_empty_axis(axes[0, 1])
plot_clv(axes[0, 2], clvs, 0)
plot_clv(axes[0, 3], clvs, 1)
setup_empty_axis(axes[0, 4])

# Row 2
plot_glider(axes[1, 0], glider)
plot_clv(axes[1, 1], clvs, 2)
plot_clv(axes[1, 2], clvs, 3)
plot_clv(axes[1, 3], clvs, 4)
plot_clv(axes[1, 4], clvs, 5)

# Row 3
setup_empty_axis(axes[2, 0])
plot_clv(axes[2, 1], clvs, 6)
plot_clv(axes[2, 2], clvs, 7)
plot_clv(axes[2, 3], clvs, 8)
plot_clv(axes[2, 4], clvs, 9)

plt.tight_layout()
plt.savefig('clv_plots.png', dpi=300)
