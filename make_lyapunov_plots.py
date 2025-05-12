import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


from torch.func import jvp, vmap
def _push_forward(f, x: torch.Tensor, V: torch.Tensor, dt):
    def inner(v_col):
        # Single-direction JVP: returns only the JVP, we ignore primal_out.
        _, jv = jvp(f, (x,), (v_col,))
        return v_col + dt * jv          # n-vector

    # V.T has shape (k, n)  |→  k independent JVPs in parallel
    return vmap(inner)(V.T).T           # back to (n, k)

def run_lyapunov(f, x0, V0, dt, steps, *, hold_at_origin=False, clv_steps=1000, return_vectors=False):
    x = x0.clone().detach()
    V = V0.clone().detach()  # current GS basis  (k columns)
    k = V.shape[1]

    S = torch.zeros((steps, k), dtype=torch.float64, device=x.device)  # log diag(R)
    Rs = []  # store QR data

    x_m = torch.zeros_like(x0)
    V_m = torch.zeros_like(V0)
    m = steps - clv_steps - 1

    if hold_at_origin:
        h = int(np.sqrt(x.shape[0]).round())
        i, j = torch.meshgrid(
            torch.arange(h, device=x.device),
            torch.arange(h, device=x.device),
            indexing="ij",
        )
        i_idx, j_idx = i.flatten(), j.flatten()

    # ---------- forward QR pass ----------
    for n in tqdm(range(steps), leave=False, desc="forward"):
        # optional translation of origin
        if hold_at_origin:
            sA = torch.sum(x) + 1e-30
            mi = torch.sum(i_idx * x) / sA
            mj = torch.sum(j_idx * x) / sA
            sh = [h // 2 - int(mi), h // 2 - int(mj)]
            x = torch.roll(x.reshape(h, h), sh, (0, 1)).flatten()
            V = torch.roll(V.reshape(h, h, -1), sh, (0, 1)).reshape_as(V)
        
        # --- push tangent basis forward ---
        # x.requires_grad_(True)
        W = _push_forward(f, x, V, dt)
        # x = x.detach()

        # --- QR orthonormalisation ---
        Q, R = torch.linalg.qr(W, mode="reduced")
        V = Q  # next GS basis
        
        # --- integrate the state ---
        x = x + dt * f(x)

        if n == m:
            x_m = x.clone()
            V_m = V.clone()
        if n > m:
            Rs.append(R)

        S[n] = torch.log(torch.abs(torch.diagonal(R)))

    # ---------- Lyapunov exponents ----------
    half = steps // 2
    cum = S.cumsum(0)
    counts = torch.arange(1, steps + 1, device=S.device, dtype=S.dtype).unsqueeze(1)
    lambda_history = cum / (counts * dt)
    lambda_history[half:] = (cum[half:] - cum[:-half]) / (half * dt)

    lambdas = lambda_history[-1]
    order = torch.argsort(lambdas, descending=True)

    results = [lambdas[order], lambda_history[:,order], x_m]
    if return_vectors:
        # ---------- GS orthonormal basis ----------
        GSV_m = V_m.clone()  # GS basis at final step

        # ---------- CLV reconstruction  ----------
        C = torch.eye(k, dtype=torch.float64, device=x.device)   # start at t_m
        for i, R in enumerate(reversed(Rs)):
            C = torch.linalg.solve_triangular(R, C, upper=True)  # C <- R^{-1}C
            if i % 100 == 0:
                C = C / torch.linalg.norm(C, dim=0, keepdim=True)  # normalise

        CLV_m = (GSV_m @ C)[:, order].clone()  # CLVs at final step
        CLV_m = CLV_m / torch.linalg.norm(CLV_m, dim=0, keepdim=True)


        results.append(GSV_m[:,order]) # GSV basis at m_th step
        results.append(CLV_m[:,order]) # CLV basis at m_th step

    return tuple(results)

# Kaplan-Yorke dimension
def KY_dim(lam):
    lam = np.where(np.abs(lam) < 1e-3, 0.0, lam)
    lam = np.sort(np.asarray(lam))[::-1]
    c = np.cumsum(lam)
    if np.all(c >= 0):
        return float(len(lam))
    j = np.where(c >= 0)[0][-1]
    return (j + 1) + c[j] / abs(lam[j + 1])


#======================== Lenia functions ========================== #
def poly_kernel(r):
    return (4*r*(1-r))**4

def build_kernel(b, R, H):
    B = np.array(b.size)
    mid = H // 2
    Dx, Dy = np.meshgrid(np.arange(-mid, mid), np.arange(-mid, mid), indexing='ij')
    D = np.sqrt(Dx**2 + Dy**2) / R * B

    K = (D<B) * poly_kernel(D % 1) * b[np.minimum(D.astype(np.int64), B-1)]
    K = K / np.sum(K)
    K_fft = np.fft.fft2(np.fft.fftshift(K))
    return K_fft

def poly_growth(U,m,s):
    G = torch.maximum(torch.tensor(0), 1 - ((U-m)/s)**2 / 9 )**4 * 2 - 1
    return (G+1)/2

def convK(A, K_fft):
    return torch.fft.ifft2(torch.fft.fft2(A) * K_fft).real

def com(A):
    device = A.device
    H, W = A.shape[-2:]
    i, j = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    i, j = i.to(device), j.to(device)
    sA = torch.sum(A, dim=(-2,-1)) + 1e-8
    mi = torch.sum(i * A ,dim=(-2,-1)) / sA
    mj = torch.sum(j * A, dim=(-2,-1)) / sA
    return mi, mj

@torch.jit.script
def simulate(A, K_fft, m, s, steps:int, hold_at_origin:bool =False):
    H, W = A.shape[-2:]
    T = torch.tensor(10., device=A.device)
    for _ in range(steps):
        A = A + (poly_growth(convK(A,K_fft),m=m,s=s) - A)/T
        if hold_at_origin:
            mi, mj = com(A)
            A = torch.roll(A, [H//2 - int(mi), W//2 - int(mj)], dims=(0, 1))
    return A

#=================================================================== #

def calculate_LEs(A, params, device='cuda'):    
    scale = 3
    H, W = (256, 256)

    b = np.array(params['b'])
    R = params['R'] * scale
    m = params['m']
    s = params['s']
    T = params['T']

    A = torch.tensor(glider, dtype=torch.float32).cuda()
    K_fft = build_kernel(b, R, H)

    K_fft = torch.tensor(K_fft, dtype=torch.complex64).to(device)
    m = torch.tensor(m, dtype=torch.float32).to(device)
    s = torch.tensor(s, dtype=torch.float32).to(device)

    def lenia_step(X):
        A = X.reshape(H, W)
        return (poly_growth(convK(A,K_fft),m,s) - A).flatten()

    k = 10
    H, W = A.shape
    X0 = A.flatten().cuda()
    V0 = torch.randn(X0.numel(), k, dtype=torch.float64)
    Q, _ = torch.linalg.qr(V0)
    V0 = Q.cuda()
    dt = 0.1

    lyapunov_exponents, histories, A_final = run_lyapunov(lenia_step, X0, V0, dt, 100000, clv_steps=5000, hold_at_origin=True, return_vectors=False)
    # lyapunov_exponents, histories, A_final, GS, CLV = run_lyapunov(lenia_step, X0, V0, dt, 100000, clv_steps=5000, hold_at_origin=True, return_vectors=True)
    lyapunov_exponents = lyapunov_exponents.cpu().detach().numpy()
    histories = histories.cpu().detach().numpy()
    print("Lyapunov exponents:", lyapunov_exponents)
    print("Kaplan-Yorke dimension:", KY_dim(lyapunov_exponents))
    return lyapunov_exponents, histories



if __name__ == "__main__":
    # gliders_list = [
    #     'glider',
    #     'glider_periodic',
    #     'glider_chaotic',
    #     'rotator',
    # ]
    # params_list = [
    #     {"R":18,"T":10,"b":[1,1/2,1/2,1],"m":0.24,"s":0.02,"kn":1,"gn":1}, # soliton
    #     {"R":18,"T":10,"b":[1,1/2,1/2,1],"m":0.2,"s":0.013,"kn":1,"gn":1}, # periodic
    #     {"R":18,"T":10,"b":[1,0.01,0.25,1],"m":0.175,"s":0.0135,"kn":1,"gn":1}, # chaotic
    #     {"R":18,"T":10,"b":[1,0.01,0.5,1],"m":0.22,"s":0.026,"kn":1,"gn":1}, # rotator
    # ]

    # output_data = []
    # for glider_name, params in zip(gliders_list, params_list):
    #     glider = np.load(glider_name+".npy")
    #     LEs, histories = calculate_LEs(glider, params)
    #     output_data.append({'name': glider_name, 'lyapunov_exponents': LEs.tolist(), 'histories': histories.tolist()})

    # import json
    # with open('lyapunov_exponents.json', 'w') as f:
    #     json.dump(output_data, f, indent=4)


    # ===== PLOTS ====== #

    import json
    with open('lyapunov_exponents.json', 'r') as f:
        data = json.load(f)

    # rotator first
    data = [data[3],data[0],data[1], data[2]]
    name_map = {
        'glider': 'Soliton',
        'glider_periodic': 'Periodic soliton',
        'glider_chaotic': 'Chaotic soliton',
        'rotator': 'Rotator'
    }
    k = 10
    row_infos = []
    fig, axs = plt.subplots(4, 2, figsize=(15, 14))
    for (le_ax, t_ax), glider_data in zip(axs, data):
        name = glider_data['name']
        title = name_map[name]
        LEs = np.array(glider_data['lyapunov_exponents'])
        D_KY = KY_dim(LEs)
        histories = np.array(glider_data['histories'])
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

    # for idx, txt in enumerate(row_infos):
    #     fig.text(0.45, 1.0 - idx * 0.25, txt, ha='center', va='bottom', fontsize=16)
    # plt.tight_layout(h_pad=2.0)
    for idx, txt in enumerate(row_infos):
        fig.text(0.45, 0.94 - idx * 0.237, txt, ha='center', va='bottom', fontsize=16, weight='bold')

    # plt.subplots_adjust(top=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.94], h_pad=4.0)

    plt.savefig('lyapunov_exponents.png', dpi=300)



