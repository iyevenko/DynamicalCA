import numpy as np
import matplotlib.pyplot as plt


# Kaplan Yorke dimension
def KY_dim(les):
    les = np.where(np.abs(les) < 1e-3, 0.0, les)  # round near zero exponents
    les = np.sort(np.asarray(les))[::-1]
    c = np.cumsum(les)
    if np.all(c >= 0):
        return float(len(les))
    j = np.where(c >= 0)[0][-1]
    return (j + 1) + c[j] / abs(les[j + 1])


names = [
    'rotator',
    'soliton',
    'periodic_soliton',
    'chaotic_soliton',
]

k = 10
row_infos = []
fig, axs = plt.subplots(4, 2, figsize=(15, 14))
for (le_ax, t_ax), name in zip(axs, names):
    title = name.replace('_', ' ').capitalize()
    LEs = np.loadtxt(f'{name}_les.txt')
    D_KY = KY_dim(LEs)
    histories = np.load(f'{name}_les_histories.npy')
    row_infos.append(title+":  $D_{KY}="+f" {D_KY:.2f}"+"$")

    le_ax.plot(np.arange(k) + 1, LEs, 'o-')
    le_ax.set_xticks(np.arange(k) + 1, [f"$\\lambda_{{{i+1}}}$" for i in range(k)])
    le_ax.set_ylabel('Lyapunov Exponent')
    le_ax.set_title('Lyapunov Spectrum')
    le_ax.grid(True, which='both')

    cmap = plt.cm.tab10
    colors = cmap(np.linspace(0, 1, histories.shape[1]))
    for i in range(histories.shape[1]):
        mantissa, exponent = f"{LEs[i]:.2e}".split('e')
        label = f"$\\lambda_{{{i+1}}}={mantissa}\\times10^{{{int(exponent)}}}$"
        t_ax.plot(np.arange(len(histories)//100)*100,histories[::100, i], color=colors[i], alpha=0.7, label=label)
    lmin = LEs[-1] * 1.5
    lmax = LEs[0] * 1.5 if LEs[0] > 1e-3 else 0.05
    t_ax.set_ylim(lmin, lmax)
    t_ax.set_xlabel('Time Step')
    t_ax.set_ylabel('Exponent Value')
    t_ax.set_title(f'Top {k} Lyapunov Exponents Over Time')
    t_ax.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left')
    t_ax.grid(True)


for idx, txt in enumerate(row_infos):
    fig.text(0.45, 0.94 - idx * 0.237, txt, ha='center', va='bottom', fontsize=16, weight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.94], h_pad=4.0)

plt.savefig('lyapunov_exponents.png', dpi=300)



