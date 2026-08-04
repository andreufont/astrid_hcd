import numpy as np
import h5py
import argparse
import scipy.stats as spt

def get_box_geometry(grid, L_Mpc, verbose=True):
    """ Figure out description of the 3D box for input grid of skewers. """

    # get box geometry
    n_xyz=grid.shape
    n_xy=int(np.sqrt(n_xyz[0]))
    n_z=n_xyz[1]
    if verbose:
        print('n_xy={}, n_z={}'.format(n_xy,n_z))
    d_xy = L_Mpc / n_xy
    d_z = L_Mpc / n_z

    # collect results to return
    results={'n_xy':n_xy,'n_z':n_z,'d_xy':d_xy,'d_z':d_z, 'L_Mpc':L_Mpc}

    # specify wavenumbers in box
    k_xy = np.fft.fftfreq(n_xy, d=d_xy) * 2. * np.pi
    k_z = np.fft.fftfreq(n_z, d=d_z) * 2. * np.pi
    # construct 3D grid of wavenumbers
    box_kx = k_xy[:,np.newaxis,np.newaxis]
    box_ky = k_xy[np.newaxis,:,np.newaxis]
    box_kz = k_z[np.newaxis,np.newaxis,:]
    box_k = np.sqrt(box_kx**2 + box_ky**2 + box_kz**2)
    # construct mu in two steps, without NaN warnings
    box_mu = box_kz/np.ones_like(box_k)
    box_mu[box_k>0.] /= box_k[box_k>0.]
    box_mu[box_k == 0.] = np.nan

    results['box_k']=box_k
    results['box_mu']=box_mu

    return results
    

def measure_p3d_Mpc(deltas1, deltas2, z, box, n_k_bins=20, k_Mpc_max=20.0, n_mu_bins=16, mask=None, verbose=True):
    
    """ Compute 3D power spectrum of input grid of skewers. 
    
    Input
    ----------
    deltas1: n-array
        Array of shape (Nsk*Nsk, Np) containing the flux deltas of field 1
    deltas2: n-array
        Array of shape (Nsk*Nsk, Np) containing the flux deltas of field 2
    z: float
        Redshift value
    box: list
        output of get_box_geometry function 
    n_k_bins: int
        Number of k bins
    k_Mpc_max: int
        Max k (in 1/Mpc) to compute P3D
    n_mu_bins: int
        Number of mu bins  
    mask: n-array, optional
        Mask to select only certain skewers (same shape as deltas)


    Output
    -----------
    results: object
        z, n_xy, n_z, d_xy, d_z, k_Mpc, mu, counts, p3d_Mpc
    """
    
    # collect relevant information 
    results={'z':z,
                'n_xy':box['n_xy'],'n_z':box['n_z'],
                'd_xy':box['d_xy'],'d_z':box['d_z']}


    # Reshape deltas
    Nsk = int(np.sqrt(deltas1.shape[0]))
    Np = deltas1.shape[1]
    deltas1, deltas2 = deltas1.reshape([Nsk, Nsk, Np]), deltas2.reshape([Nsk, Nsk, Np])
    if mask is not None:
        mask = mask.reshape([Nsk, Nsk])
    
    # get Fourier modes
    if mask is not None:
        deltas1[~mask, :] = 0
        deltas2[~mask, :] = 0
        f_valid = np.mean(mask)
        if verbose:
            print('Masking applied')
    else:
        f_valid = 1
        
    norm_fac = 1.0 / deltas1.size
    modes1 = np.fft.fftn(deltas1) * norm_fac
    modes2 = np.fft.fftn(deltas2) * norm_fac
    
    # get raw power
    raw_p3d = (modes1*modes2.conjugate()).real
    results['p3d_raw'] = raw_p3d    

    # change format and get rid of first mode (k=0)
    k_box = box['box_k'].flatten()[1:]
    mu_box = np.absolute(box['box_mu']).flatten()[1:]
    p3d_box = raw_p3d.flatten()[1:]


    # define k-binning (in 1/Mpc)
    lnk_max = np.log(k_Mpc_max)
    # set minimum k to make sure we cover fundamental mode
    lnk_min = np.log(0.9999*np.min(k_box[k_box > 0.]))
    lnk_bin_max = lnk_max + (lnk_max-lnk_min)/(n_k_bins-1)
    lnk_bin_edges = np.linspace(lnk_min,lnk_bin_max,n_k_bins+1)
    k_bin_edges = np.exp(lnk_bin_edges)
    # define mu-binning
    mu_bin_edges = np.linspace(0., 1., n_mu_bins + 1)

    results['k_bin_edges'] = k_bin_edges
    results['mu_bin_edges'] = mu_bin_edges
    results['k_center'] = 0.5 * (k_bin_edges[:-1] + k_bin_edges[1:])
    results['mu_center'] = 0.5 * (mu_bin_edges[:-1] + mu_bin_edges[1:])

    # compute bin averages (not including first mode with k=0, mu=0)
    binned_p3d = spt.binned_statistic_2d(k_box, mu_box, p3d_box,
                statistic = 'mean', bins = [k_bin_edges,mu_bin_edges])[0]
    binned_counts = spt.binned_statistic_2d(k_box,mu_box,p3d_box,
            statistic='count', bins=[k_bin_edges,mu_bin_edges])[0]
    # compute binned values for (k,mu)
    binned_k = spt.binned_statistic_2d(k_box, mu_box, k_box,
            statistic = 'mean', bins = [k_bin_edges,mu_bin_edges])[0]
    binned_mu = spt.binned_statistic_2d(k_box, mu_box, mu_box,
                    statistic = 'mean', bins = [k_bin_edges,mu_bin_edges])[0]

    # convert to list 
    results['k_Mpc'] = binned_k.tolist()
    results['mu'] = binned_mu.tolist()
    results['counts'] = binned_counts.tolist()
    # quantity above is dimensionless, multiply by box size (in Mpc)
    results['p3d_Mpc'] = (binned_p3d * box['L_Mpc']**3).tolist()

    return results
    

def main(args):
    
    # Reading delta file attributes
    filename = args.deltas
    with h5py.File(filename, 'r') as f:
            
        # Atributes
        logNHi_min = f.attrs['logNHI_min']
        logNHi_max = f.attrs['logNHI_max']
        smth_factor = f.attrs['Smoothing factor']
        Lbox = f.attrs['box_size_Mpch']
        Nsk = f.attrs['skewers_per_side']
        Np = f.attrs['pixels_per_skewer']
        Pw = f.attrs['pixel_width_Mpch']
        Ssk = f.attrs['skewer_separation_Mpch']

        Lbox_Mpc = Lbox/args.hubble
        Pw_Mpc = Pw/args.hubble
        Ssk_Mpc = Ssk/args.hubble

        C = f['C'][()]
    

    # Total 
    # Reading delta files
    with h5py.File(filename, 'r') as f:
        delta_tot = f['deltas/delta_tot'][:]
        if args.mask:
            mask = f['Mask'][()][:]
        else:
            mask = None

    # Box info (this is the same for the three fields)
    box = get_box_geometry(delta_tot, Lbox_Mpc)
    k_Nyq = np.pi*np.min([Np, Nsk])/Lbox_Mpc
    if args.k_max:
        k_max = args.k_max
    else:
        k_max = k_Nyq/2

    # Computing P3d

    # Total
    results = measure_p3d_Mpc(delta_tot, delta_tot, args.z, box, n_k_bins=args.n_k_bins, 
                              k_Mpc_max=k_max, n_mu_bins=args.n_mu_bins, mask=mask)
    p3d_tot = np.array(results['p3d_Mpc'])
    counts_tot = np.array(results['counts'])
    k = np.array(results['k_Mpc'])
    mu = np.array(results['mu'])
    mu_center = np.array(results['mu_center'])
    mu_bins = np.array(results['mu_bin_edges'])
    del delta_tot, results

    
    # Lya
    # Reading delta files
    with h5py.File(filename, 'r') as f:
        delta_lya = f['deltas/delta_lya'][:]

    # Computing P3d
    results = measure_p3d_Mpc(delta_lya, delta_lya, args.z, box, n_k_bins=args.n_k_bins, 
                              k_Mpc_max=k_max, n_mu_bins=args.n_mu_bins, mask=mask, verbose=False)
    p3d_lya = np.array(results['p3d_Mpc'])
    counts_lya = np.array(results['counts'])
    del results

    
    # Hcd
    # Reading delta files
    with h5py.File(filename, 'r') as f:
        delta_hcd = f['deltas/delta_hcd'][:]

    # Computing P3d
    results = measure_p3d_Mpc(delta_hcd, delta_hcd, args.z, box, n_k_bins=args.n_k_bins, 
                              k_Mpc_max=k_max, n_mu_bins=args.n_mu_bins, mask=mask, verbose=False)
    p3d_hcd = np.array(results['p3d_Mpc'])
    counts_hcd = np.array(results['counts'])
    del results

    
    # LyaxHCD
    # Computing P3d
    results = measure_p3d_Mpc(delta_lya, delta_hcd, args.z, box, n_k_bins=args.n_k_bins, 
                              k_Mpc_max=k_max, n_mu_bins=args.n_mu_bins, mask=mask, verbose=False)
    p3d_lyahcd = np.array(results['p3d_Mpc'])
    counts_lyahcd = np.array(results['counts'])
    del results


    # 3Lya
    delta_lyahcd = delta_lya*delta_hcd
    # Computing P3d
    results = measure_p3d_Mpc(delta_lyahcd, delta_lya, args.z, box, n_k_bins=args.n_k_bins, 
                              k_Mpc_max=k_max, n_mu_bins=args.n_mu_bins, mask=mask, verbose=False)
    p3d_3lya = np.array(results['p3d_Mpc'])
    counts_3lya = np.array(results['counts'])
    del delta_lya, results


    # 3HCD
    # Computing P3d
    results = measure_p3d_Mpc(delta_lyahcd, delta_hcd, args.z, box, n_k_bins=args.n_k_bins, 
                              k_Mpc_max=k_max, n_mu_bins=args.n_mu_bins, mask=mask, verbose=False)
    p3d_3hcd = np.array(results['p3d_Mpc'])
    counts_3hcd = np.array(results['counts'])
    del delta_hcd, results

    # 4
    # Computing P3d
    results = measure_p3d_Mpc(delta_lyahcd, delta_lyahcd, args.z, box, n_k_bins=args.n_k_bins, 
                              k_Mpc_max=k_max, n_mu_bins=args.n_mu_bins, mask=mask, verbose=False)
    p3d_4 = np.array(results['p3d_Mpc'])
    counts_4 = np.array(results['counts'])
    del delta_lyahcd, results
        
    
    # Writing the files out
    with h5py.File(f'{args.output_file}.hdf5', 'w') as f:
    
        f.attrs['logNHI_min'] = logNHi_min
        f.attrs['logNHI_max'] = logNHi_max
        f.attrs['Smoothing factor'] = smth_factor
        f.attrs['Box_size_Mpch'] = Lbox
        f.attrs['skewers_per_boxside'] = Nsk
        f.attrs['pixels_per_skewer'] = Np
        f.attrs['pixel_width_Mpch'] = Pw
        f.attrs['skewer_separation_Mpch'] = Ssk

        for m in f.attrs.keys():
            print(f'{m} = {f.attrs[m]}')
            
        f.create_dataset('k', data=k)  # Inverse units of Lbox (1/Mpc)
        f.create_dataset('mu_bins', data=mu_bins)  
        f.create_dataset('mu_center', data=mu_center)
        f.create_dataset('mu', data=mu)
        f.create_dataset('C', data=C)
        
        grp1 = f.create_group('p3ds')
        grp1.create_dataset('p3d_tot', data=p3d_tot)  # Same units as Lbox^3 (Mpc)^3
        grp1.create_dataset('p3d_lya', data=p3d_lya)
        grp1.create_dataset('p3d_hcd', data=p3d_hcd)
        grp1.create_dataset('p3d_lyahcd', data=p3d_lyahcd)
        grp1.create_dataset('p3d_3lya', data=p3d_3lya)
        grp1.create_dataset('p3d_3hcd', data=p3d_3hcd)
        grp1.create_dataset('p3d_4', data=p3d_4)

        grp2 = f.create_group('counts')
        grp2.create_dataset('counts_tot', data=counts_tot)
        grp2.create_dataset('counts_lya', data=counts_lya)
        grp2.create_dataset('counts_hcd', data=counts_hcd)
        grp2.create_dataset('counts_lyahcd', data=counts_lyahcd)
        grp2.create_dataset('counts_3lya', data=counts_3lya)
        grp2.create_dataset('counts_3hcd', data=counts_3hcd)
        grp2.create_dataset('counts_4', data=counts_4)
        
    print(f'Results saved in {args.output_file}.hdf5')

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description= "Compute the different contributions to P3D given the deltas and mean flux of the miniboxes the simulation has been divided into")
    
    parser.add_argument("--deltas", type=str, required=True,
                        help="HDF5 file with the deltas of the different fields")
    parser.add_argument("--z", type=float, required=True,
                        help="Redshift")
    parser.add_argument("--hubble", type=float, required=True,
                        help="Hubble parameter")
    parser.add_argument("--n_k_bins", type=int, default=20,
                        help="Number of k bins")
    parser.add_argument("--k_max", type=float, default=None,
                        help="Max k value (inverse units of Lbox)")
    parser.add_argument("--n_mu_bins", type=int, default=16,
                        help="Number of mu bins")
    parser.add_argument("--mask", action='store_true',
                        help="Whether the user wants to mask certain skewers (based on their logNHi value) or not. Default is not")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to HDF5 file where the results will be stored")

    args = parser.parse_args()

    main(args)
