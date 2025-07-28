import h5py
import numpy as np
import os
import re
import pickle
from matplotlib import pyplot as plt

def plot_Px_per_rbin(Px_means, Px_stds, Px_model, Px_model_std, k):
    for rbin in Px_means:
        plt.figure(figsize=(8, 5))

        # Plot all Px_* with shading
        for key in Px_means[rbin]:
            mean = Px_means[rbin][key]
            std = Px_stds[rbin][key]
            label = key.replace('Px_', r'$P_{\times}^{') + '}$'
            plt.plot(k, mean, label=label)
            plt.fill_between(k, mean - std, mean + std, alpha=0.3)

        # Plot model
        model = Px_model[rbin]
        model_std = Px_model_std[rbin]
        plt.plot(k, model, linestyle='--', label=r'$P_{\times}^{\mathrm{model}}$')
        plt.fill_between(k, model - model_std, model + model_std, alpha=0.3, color='gray')

        plt.xscale('log')
        plt.xlim(k[1], 6)
        plt.xlabel(r'$k$')
        plt.ylabel(r'$P_{\times}(k)$')
        plt.title(f'Cross Power Spectrum - {rbin}')
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_all_P1D_with_stds(k, P1D_means, P1D_stds, P1D_model, P1D_model_std, kmin, kmax):
    """
    Plots all P1D_* components with their standard deviation bands,
    plus the model with its std deviation.

    Parameters:
        k (np.array): Wavenumber array
        P1D_means (dict): Mean values of each P1D component
        P1D_stds (dict): Standard deviations of each P1D component
        P1D_model (np.array): Total model (sum of all but P1D_F)
        P1D_model_std (np.array): Std deviation of total model
        kmin (float): Lower limit for x-axis
        kmax (float): Upper limit for x-axis
    """
    plt.figure(figsize=(8, 6))

    # Define plotting order and styles
    components = ['P1D_F', 'P1D_a', 'P1D_H', 'P1D_aH', 'P1D_a3', 'P1D_H3', 'P1D_p4']
    colors = {
        'P1D_F': 'tab:blue',
        'P1D_a': 'tab:orange',
        'P1D_H': 'tab:green',
        'P1D_aH': 'tab:red',
        'P1D_a3': 'tab:purple',
        'P1D_H3': 'tab:brown',
        'P1D_p4': 'tab:pink',
    }
    labels = {
        'P1D_F': r'$P_{1D}^F$',
        'P1D_a': r'$P_{1D}^H$',
        'P1D_H': r'$P_{1D}^a$',
        'P1D_aH': r'$P_{1D}^{aH}$',
        'P1D_a3': r'$P_{1D}^{a3}$',
        'P1D_H3': r'$P_{1D}^{H3}$',
        'P1D_p4': r'$P_{1D}^{p4}$',
    }

    # Plot each component with std dev band
    for comp in components:
        if comp in P1D_means:
            mean = P1D_means[comp]
            std = P1D_stds.get(comp, np.zeros_like(mean))
            plt.plot(k, mean, label=labels.get(comp, comp), color=colors.get(comp, None))
            plt.fill_between(k, mean - std, mean + std, color=colors.get(comp, None), alpha=0.2)

    # Plot total model
    plt.plot(k, P1D_model, color='black', linestyle='--', label=r'$P_{1D}^{\mathrm{model}}$')
    plt.fill_between(k, P1D_model - P1D_model_std, P1D_model + P1D_model_std, color='gray', alpha=0.3)

    plt.title(r'Average $P_{1D}(k_\parallel)$ for all sub boxes')
    plt.xlabel(r'$k_\parallel$ [h/Mpc]')
    plt.ylabel(r'$P_{1D}(k_\parallel)$ [Mpc/h]')
    plt.xscale('log')
    plt.xlim(kmin if kmin is not None else k[1], kmax)
    plt.legend()
    plt.tight_layout()
    plt.show()

def avg_std_Px(Px_all, block_grid_size, C, Np):
    Px_means = {}
    Px_stds = {}
    Px_model = {}
    Px_model_std = {}

    suffixes = [f"_{i}{j}" for i in range(block_grid_size) for j in range(block_grid_size)]

    for rbin, Px_dict in Px_all.items():
        Px_means[rbin] = {}
        Px_stds[rbin] = {}

        Px_sum = np.zeros(Np)
        Px_sum_var = np.zeros(Np)

        prefixes = set(key[:-3] for key in Px_dict.keys())

        for prefix in prefixes:
            arrays = []
            for suffix in suffixes:
                key = prefix + suffix
                if key in Px_dict:
                    arrays.append(Px_dict[key])
            if arrays:
                stacked = np.stack(arrays)
                mean = np.mean(stacked, axis=0)
                std = np.std(stacked, axis=0)

                Px_means[rbin][prefix] = mean
                Px_stds[rbin][prefix] = std

                if prefix != 'Px_F':
                    Px_sum += mean
                    Px_sum_var += std**2

        model = Px_sum / (1 + C)**2
        model_var = Px_sum_var / (1 + C)**4
        model_std = np.sqrt(model_var)

        Px_model[rbin] = model
        Px_model_std[rbin] = model_std

    return Px_means, Px_stds, Px_model, Px_model_std


def avg_std_P1D(P1D_all, block_grid_size, C, Np):
    P1D_means = {}
    P1D_stds = {}

    P1D_sum = np.zeros(Np)
    P1D_sum_var = np.zeros(Np)

    prefixes = set(key[:-3] for key in P1D_all.keys())
    suffixes = [f"_{i}{j}" for i in range(block_grid_size) for j in range(block_grid_size)]

    for prefix in prefixes:
        arrays = []
        for suffix in suffixes:
            key = prefix + suffix
            if key in P1D_all:
                arrays.append(P1D_all[key])
        if arrays:
            stacked = np.stack(arrays)
            mean = np.mean(stacked, axis=0)
            std = np.std(stacked, axis=0)

            P1D_means[prefix] = mean
            P1D_stds[prefix] = std

            if prefix != 'P1D_F':
                P1D_sum += mean
                P1D_sum_var += std**2

    P1D_model = P1D_sum / (1 + C)**2
    P1D_model_var = P1D_sum_var / (1 + C)**4
    P1D_model_std = np.sqrt(P1D_model_var)

    return P1D_means, P1D_stds, P1D_model, P1D_model_std



def load_power_spectra(load_path):
    """
    Load P1D_all and Px_all dictionaries from the specified directory.

    Parameters:
    - load_path: str, path to directory containing 'P1D_all.pkl' and 'Px_all.pkl'

    Returns:
    - P1D_all: dict
    - Px_all: dict
    """
    p1d_file = os.path.join(load_path, "P1D_all.pkl")
    px_file = os.path.join(load_path, "Px_all.pkl")

    with open(p1d_file, "rb") as f:
        P1D_all = pickle.load(f)

    with open(px_file, "rb") as f:
        Px_all = pickle.load(f)

    return P1D_all, Px_all


def save_power_spectra(P1D_all, Px_all, save_path):
    """
    Save P1D_all and Px_all dictionaries to files in the specified directory.

    Parameters:
    - P1D_all: dict
    - Px_all: dict
    - save_path: str, path to directory where files will be saved

    Output:
    - Saves 'P1D_all.pkl' and 'Px_all.pkl' to save_path
    """
    os.makedirs(save_path, exist_ok=True)

    p1d_file = os.path.join(save_path, "P1D_all.pkl")
    px_file = os.path.join(save_path, "Px_all.pkl")

    with open(p1d_file, "wb") as f:
        pickle.dump(P1D_all, f)

    with open(px_file, "wb") as f:
        pickle.dump(Px_all, f)

    print(f"Saved P1D_all to {p1d_file}")
    print(f"Saved Px_all to {px_file}")


def flux(tau_tot, tau_lya, tau_hcd):
    flux_tot = np.exp(-tau_tot)
    flux_lya = np.exp(-tau_lya)
    flux_hcd = np.exp(-tau_hcd)
    return flux_tot, flux_lya, flux_hcd

def weighted_P1D_average(P1D_all, tau_tot, tau_lya, tau_hcd, block_grid_size, Np):
    """
    Compute weighted average of P1D spectra based on flux product weights.

    Parameters:
    - P1D_all: dict, keys like 'P1D_F_00', 'P1D_a_12', values: arrays length Np
    - tau_tot, tau_lya, tau_hcd: 2D arrays shape (block_grid_size, block_grid_size)
    - block_grid_size: int, number of blocks per side
    - Np: int, length of P1D arrays

    Returns:
    - weighted_dict: dict with keys = unique P1D fields (P1D_F, P1D_a, etc.)
                     values = weighted average arrays length Np
    """
    F_tot, F_lya, F_hcd = flux(tau_tot, tau_lya, tau_hcd)
    pattern = re.compile(r"^(P1D_\w+)_([0-4])([0-4])$")

    fields = set()
    for key in P1D_all.keys():
        m = pattern.match(key)
        if m:
            fields.add(m.group(1))

    n_blocks = block_grid_size**2
    weighted_dict = {}

    # Precompute mean fluxes over all blocks (for normalization)
    mean_F_tot = np.mean(F_tot)
    mean_F_lya = np.mean(F_lya)
    mean_F_hcd = np.mean(F_hcd)

    for field in fields:
        weighted_sum = np.zeros(Np)

        for i in range(block_grid_size):
            for j in range(block_grid_size):
                key = f"{field}_{i}{j}"
                if key not in P1D_all:
                    raise KeyError(f"Missing key in P1D_all: {key}")

                P = P1D_all[key]

                # Determine weights based on field type
                Ft = F_tot[i, j]
                Fa = F_lya[i, j]
                Fh = F_hcd[i, j]

                if field == "P1D_F":
                    weight = Ft * Ft
                    norm = mean_F_tot * mean_F_tot
                elif field == "P1D_a":
                    weight = Fa * Fa
                    norm = mean_F_lya * mean_F_lya
                elif field == "P1D_H":
                    weight = Fh * Fh
                    norm = mean_F_hcd * mean_F_hcd
                elif field == "P1D_aH":
                    weight = Fa * Fh
                    norm = mean_F_lya * mean_F_hcd
                elif field == "P1D_a3":
                    weight = Fa * Fa * Fh
                    norm = (mean_F_lya ** 2) * mean_F_hcd
                elif field == "P1D_H3":
                    weight = Fa * Fh * Fh
                    norm = mean_F_lya * (mean_F_hcd ** 2)
                elif field == "P1D_p4":
                    weight = Fa * Fa * Fh * Fh
                    norm = (mean_F_lya ** 2) * (mean_F_hcd ** 2)
                else:
                    raise ValueError(f"Unknown field: {field}")

                weighted_sum += weight * P

        weighted_avg = weighted_sum / n_blocks / norm
        weighted_dict[field] = weighted_avg

    return weighted_dict

def compute_C(tau_tot_blocks, tau_lya_blocks, tau_hcd_blocks, block_grid_size):
    constant = np.empty((block_grid_size, block_grid_size))
    flux_tot_map = np.empty_like(constant)
    flux_lya_map = np.empty_like(constant)
    flux_hcd_map = np.empty_like(constant)

    for i in range(block_grid_size):
        for j in range(block_grid_size):
            tau_totb = tau_tot_blocks[i, j]
            tau_lyab = tau_lya_blocks[i, j]
            tau_hcdb = tau_hcd_blocks[i, j]

            # Convert to flux
            flux_tot, flux_lya, flux_hcd = flux(tau_totb, tau_lyab, tau_hcdb)
            
            # Mean fluxes
            avg_tot = np.mean(flux_tot)
            avg_lya = np.mean(flux_lya)
            avg_hcd = np.mean(flux_hcd)

            # Store block results
            constant[i, j] = avg_tot / (avg_lya * avg_hcd) - 1
            flux_tot_map[i, j] = avg_tot
            flux_lya_map[i, j] = avg_lya
            flux_hcd_map[i, j] = avg_hcd

    # Compute global averages
    avg_lya_global = np.mean(flux_lya_map, keepdims=True)
    avg_hcd_global = np.mean(flux_hcd_map, keepdims=True)

    # Weighted correction average
    weight = (flux_lya_map / avg_lya_global) * (flux_hcd_map / avg_hcd_global)
    C = np.mean(constant * weight)

    return C


def compute_p1d(tau_tot_blocks, tau_lya_blocks, tau_hcd_blocks, L_hMpc, Np, block_grid_size):
    P1D_all = {}

    for i in range(block_grid_size):
        for j in range(block_grid_size):
            print(f"Computing P1D for block ({i}, {j})")

            tau_totb = tau_tot_blocks[i, j]
            tau_lyab = tau_lya_blocks[i, j]
            tau_hcdb = tau_hcd_blocks[i, j]

            dF_tot = delta_F(tau_totb)
            dF_lya = delta_F(tau_lyab)
            dF_hcd = delta_F(tau_hcdb)
            dF_lyahcd = dF_lya * dF_hcd

            fft_tot = np.fft.fft(dF_tot)
            fft_lya = np.fft.fft(dF_lya)
            fft_hcd = np.fft.fft(dF_hcd)
            fft_lyahcd = np.fft.fft(dF_lyahcd)

            P1D_dict = {}
            P1D_dict['P1D_F'] = P1D_sum(fft_tot, fft_tot, L_hMpc, Np)
            P1D_dict['P1D_a'] = P1D_sum(fft_lya, fft_lya, L_hMpc, Np)
            P1D_dict['P1D_H'] = P1D_sum(fft_hcd, fft_hcd, L_hMpc, Np)
            P1D_dict['P1D_aH'] = P1D_sum(fft_lya, fft_hcd, L_hMpc, Np)
            tmp = P1D_sum(fft_hcd, fft_lya, L_hMpc, Np)
            P1D_dict['P1D_aH'] += tmp
            P1D_dict['P1D_a3'] = P1D_sum(fft_lya, fft_lyahcd, L_hMpc, Np)
            tmp = P1D_sum(fft_lyahcd, fft_lya, L_hMpc, Np)
            P1D_dict['P1D_a3'] += tmp
            P1D_dict['P1D_H3'] = P1D_sum(fft_hcd, fft_lyahcd, L_hMpc, Np)
            tmp = P1D_sum(fft_lyahcd, fft_hcd, L_hMpc, Np)
            P1D_dict['P1D_H3'] += tmp
            P1D_dict['P1D_p4'] = P1D_sum(fft_lyahcd, fft_lyahcd, L_hMpc, Np)

            for key, P1D in P1D_dict.items():
                dict_key = f"{key}_{i}{j}"
                P1D_all[dict_key] = P1D.real

    print("Finished computing all P1D.")
    return P1D_all


def compute_px(tau_tot_blocks, tau_lya_blocks, tau_hcd_blocks, L_hMpc, Np, block_grid_size, r_edges, Nsk):
    r_bins = len(r_edges) - 1
    Px_all = {f"rbin{b+1}": {} for b in range(r_bins)}
    block_size = Nsk // block_grid_size
    dxy_hMpc = L_hMpc / Nsk
    r_edges_pix = r_edges / dxy_hMpc

    for b in range(r_bins):
        rmin = r_edges_pix[b]
        rmax = r_edges_pix[b + 1]
        print(f"\nProcessing radial bin {b+1}: {rmin:.2f} < r < {rmax:.2f} pixels")

        for i in range(block_grid_size):
            for j in range(block_grid_size):
                print(f"  Block ({i}, {j})")

                tau_totb = tau_tot_blocks[i, j]
                tau_lyab = tau_lya_blocks[i, j]
                tau_hcdb = tau_hcd_blocks[i, j]
                N_skewers = tau_totb.shape[0]

                ix, iy = np.divmod(np.arange(N_skewers), block_size)
                dx = ix[:, None] - ix[None, :]
                dy = iy[:, None] - iy[None, :]
                distance_matrix = np.sqrt(dx**2 + dy**2)
                mask = (distance_matrix >= rmin) & (distance_matrix < rmax)

                dF_tot = delta_F(tau_totb)
                dF_lya = delta_F(tau_lyab)
                dF_hcd = delta_F(tau_hcdb)
                dF_lyahcd = dF_lya * dF_hcd

                fft_tot = np.fft.fft(dF_tot)
                fft_lya = np.fft.fft(dF_lya)
                fft_hcd = np.fft.fft(dF_hcd)
                fft_lyahcd = np.fft.fft(dF_lyahcd)

                Px_dict = {}
                Px_dict['Px_F'], _ = Px_sum(fft_tot, fft_tot, mask, L_hMpc, Np)
                Px_dict['Px_a'], _ = Px_sum(fft_lya, fft_lya, mask, L_hMpc, Np)
                Px_dict['Px_H'], _ = Px_sum(fft_hcd, fft_hcd, mask, L_hMpc, Np)
                Px_dict['Px_aH'], _ = Px_sum(fft_lya, fft_hcd, mask, L_hMpc, Np)
                tmp, _ = Px_sum(fft_hcd, fft_lya, mask, L_hMpc, Np)
                Px_dict['Px_aH'] += tmp
                Px_dict['Px_a3'], _ = Px_sum(fft_lya, fft_lyahcd, mask, L_hMpc, Np)
                tmp, _ = Px_sum(fft_lyahcd, fft_lya, mask, L_hMpc, Np)
                Px_dict['Px_a3'] += tmp
                Px_dict['Px_H3'], _ = Px_sum(fft_hcd, fft_lyahcd, mask, L_hMpc, Np)
                tmp, _ = Px_sum(fft_lyahcd, fft_hcd, mask, L_hMpc, Np)
                Px_dict['Px_H3'] += tmp
                Px_dict['Px_p4'], _ = Px_sum(fft_lyahcd, fft_lyahcd, mask, L_hMpc, Np)

                for key, Px in Px_dict.items():
                    dict_key = f"{key}_{i}{j}"
                    Px_all[f"rbin{b+1}"][dict_key] = Px.real

    print("\nFinished all radial bins.")
    return Px_all



def cd_filter(tau_on_blocks, tau_off_blocks, colden_blocks, block_grid_size, max_logN_mask):
    new_tau_on_blocks = np.empty((block_grid_size, block_grid_size), dtype=object)
    new_tau_off_blocks = np.empty((block_grid_size, block_grid_size), dtype=object)

    for i in range(block_grid_size):
        for j in range(block_grid_size):
            # Flattened skewer block shape: (block_size**2, Np)
            skewer_colden = colden_blocks[i][j]  # shape: (block_size², Np)
            max_logN_in_sk = np.log10(np.max(skewer_colden, axis=1))  # shape: (block_size²,)
            mask = max_logN_in_sk <= max_logN_mask

            new_tau_on_blocks[i][j] = tau_on_blocks[i][j][mask]
            new_tau_off_blocks[i][j] = tau_off_blocks[i][j][mask]

    return new_tau_on_blocks, new_tau_off_blocks

def apply_zero_filter(tau_on, tau_off, colden, max_logN_mask):
    """
    Sets to zero the skewers where max(log10(N)) > max_logN_mask.
    Keeps shape (Nsk², Np) intact for reshaping.
    """
    Nsk2, Np = tau_on.shape
    max_logN = np.log10(np.max(colden, axis=1))
    mask = max_logN <= max_logN_mask  # True = keep, False = zero out

    tau_on_filtered = tau_on.copy()
    tau_off_filtered = tau_off.copy()

    tau_on_filtered[~mask] = 0.0
    tau_off_filtered[~mask] = 0.0

    return tau_on_filtered, tau_off_filtered

def reshape_blocks(data, Nsk, block_grid_size, block_size, Np):
    """
    Reshapes flat data of shape (Nsk², Np) into blocks of shape
    (block_grid_size², variable # of skewers, Np),
    skipping any all-zero skewers.
    """
    data_2d = data.reshape(Nsk, Nsk, Np)
    flat_blocks = np.empty((block_grid_size, block_grid_size), dtype=object)

    for i in range(block_grid_size):
        for j in range(block_grid_size):
            r0, r1 = i * block_size, (i + 1) * block_size # Row start and end
            c0, c1 = j * block_size, (j + 1) * block_size # Column start and end

            block = data_2d[r0:r1, c0:c1, :]  # shape: (block_size, block_size, Np)
            block_flat = block.reshape(-1, Np)  # shape: (block_size², Np)

            # Remove zeroed skewers
            nonzero_mask = np.any(block_flat != 0.0, axis=1)
            filtered_block = block_flat[nonzero_mask]

            flat_blocks[i, j] = filtered_block

    return flat_blocks

def divbox(tau_tot, tau_lya, tau_hcd, Nsk, Np, block_grid_size, block_size):
    tau_tot_blocks = reshape_blocks(tau_tot, Nsk, block_grid_size, block_size, Np)
    tau_lya_blocks = reshape_blocks(tau_lya, Nsk, block_grid_size, block_size, Np)
    tau_hcd_blocks = reshape_blocks(tau_hcd, Nsk, block_grid_size, block_size, Np)

    return tau_tot_blocks, tau_lya_blocks, tau_hcd_blocks

def load_sim(fname_on, fname_off):
    with h5py.File(fname_on,'r') as f:
        tau_on = f['tau/H/1/1215'][:]
        colden = f['colden']['H/1'][:]
    with h5py.File(fname_off,'r') as f:
        tau_off = f['tau/H/1/1215'][:]
    return tau_on, tau_off, colden

def fields(tau_on, tau_off, L_hMpc, Np, r_hMpc=0.3):
    tau_max = np.fmax(tau_on, tau_off)
    tau_hcd = tau_max - tau_off
    k = np.fft.rfftfreq(Np) * 2 * np.pi / (L_hMpc / Np)
    tau_modes = np.fft.rfft(tau_hcd)
    tau_hcd = np.fft.irfft(tau_modes * np.exp(-(r_hMpc * k)**2))
    tau_tot = tau_off + tau_hcd
    return tau_tot, tau_off, tau_hcd

def delta_F(tau):
    flux = np.exp(-tau)
    avg = np.mean(flux, keepdims=True)
    return flux / avg - 1

def Px_sum(F1, F2, mask, L_hMpc, Np):
    i_idx, j_idx = np.where(np.triu(mask, k=1))
    A = F1[i_idx]
    B = F2[j_idx].conj()
    Px = np.sum(A * B, axis=0)
    npairs = len(i_idx)
    if npairs > 0:
        Px /= npairs
        Px = Px * (L_hMpc / Np**2)
    else:
        Px[:] = 0.0
    return Px, npairs

def P1D_sum(F1, F2, L_hMpc, Np):
    F = F1 * F2.conj() * (L_hMpc / Np**2)
    P1D = np.mean(F, axis=0)
    return P1D
