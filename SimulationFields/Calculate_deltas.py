import numpy as np
import h5py
import argparse

def div_box(Nbox, Nsk, Np, *arrays):
    """ This function divides an original array with shape (Nsk**2, Np) into Nbox miniboxes

    Parameters:
    -----------------
    Nbox : int
        Number of miniboxes. If the original shape is not a multiple of Nmbox, the functions gets an error.
    Nsk : int 
        Number of skewer per side in the original box..
    Np : int
        Number of pixel per skewer in the original box.
    arrays : n-array(s)
        Original array(s) that the user wants to divide.

    Returns:
    ----------------
    new_array : list
        Divided array(s) into desired shape
    """

    mb_per_side = int(np.sqrt(Nbox))  # Number of miniboxes per side
    if mb_per_side**2 != Nbox:
        raise ValueError("Nbox must be a perfect square")
    
    mb_size = Nsk // mb_per_side  # Number of skewers per side inside each 

    # Computing slices
    slices = []
    for j_mb in range(mb_per_side):  
        for i_mb in range(mb_per_side):
            imin, imax = i_mb*mb_size, (i_mb+1)*mb_size
            jmin, jmax = j_mb*mb_size, (j_mb+1)*mb_size
            slices.append((imin, imax, jmin, jmax))

    results = []
    for arr in arrays:
        
        if arr.shape != (Nsk**2, Np):
            raise ValueError(f"Expected shape {(Nsk**2, Np)}, got {arr.shape}")
            
        grid = arr.reshape(Nsk, Nsk, Np)
        new_array = np.zeros((Nbox, mb_size**2, Np))
        for mb_id, (imin, imax, jmin, jmax) in enumerate(slices):
            minibox = grid[imin:imax, jmin:jmax, :]
            new_array[mb_id] = minibox.reshape(-1, Np)

        results.append(new_array) 

    return results, mb_size


def mask_skewers(colden, logNHi_min, logNHi_max, Nsk, *arrays):
    """
    Function to mask skewers with maximum column density value larger than logNHi_max and minimiun column density value smaller than logNHi_min

    Parameters:
    --------------------
    colden: n-array
        Column density values of n skewers. Units should be in cm^-2
    logNHI_min: value
        Log(10) of minimun column density value to be considered.
    logNHI_max: value
        Log(10) of maximun column density value to be considered.
    Nsk: value
        Total number of skewers
    arrays : n-array(s)
        Original array(s) that the user wants to mask. Must be same shape as colden.


    Returns:
    --------------------
    colden_mask: n-array
        Column density values of skewers eith maximun and minimun colden values within the specified range
    new_array : list
        Masked array(s) 

    """
    
    colden_max, colden_min = np.max(colden, axis=1), np.min(colden, axis=1)
    if logNHi_min == 0:
        mask = colden_max <= 10**logNHi_max
    else:
        mask = (colden_min >= 10**logNHi_min) & (colden_max <= 10**logNHi_max)
    
    #print('Number of l.o.s eliminated:', Nsk*Nsk-mask.sum(), '(', (Nsk*Nsk-mask.sum())*100/(Nsk*Nsk), '%)')
    #print('Number of l.o.s to keep:', mask.sum())

    mask_sum = mask.sum()

    colden_mask = colden[mask]
    masked_arrays = []
    for arr in arrays:
        new_array = arr[mask]
        masked_arrays.append(new_array)

        
    return colden_mask, mask_sum, masked_arrays


def different_contributions(tau_on, tau_off, smth_factor, Np, Pw):
    """
    Function to split the optical depth into the different contributions: lya, hcd and total (sum of both)

    Parameters:
    -------------------
    tau_on: n-array
        Optical depth with damping wings considered
    tau_off: n-array
        Optical depth without damping wings
    smth_factor: value
        Smoothing factor
    Np: value
        Number of pixels per skewer
    Pw: value
        Pixel width 

    Returns:
    ------------
    tau_hcd, tau_lya, tau_tot: n-arrays
        Contributions of hcd (tau_hcd) and lyman alpha forest optical depths (tau_lya) to the total optical depth (tau_tot)

    """

    tau_max = np.maximum(tau_on, tau_off)
    tau_hcd_init = tau_max - tau_off  # only when a pixel has been influenced by an hcd it will have tau_hcd different than 0
        
    # Smoothing:
    k_smooth = np.fft.rfftfreq(Np)*2*np.pi/Pw
    tau_modes = np.fft.rfft(tau_hcd_init)
    tau_hcd = np.fft.irfft(tau_modes*(np.exp(-(smth_factor*k_smooth)**2)))

    # Contributions:
    tau_lya = tau_off
    tau_tot = tau_hcd + tau_lya

    return tau_hcd, tau_lya, tau_tot


def deltas(tau_hcd, tau_lya, tau_tot):
    """
    Function to calculate the flux deltas of lyman alpha forest, hcds and total flux

    Parameters:
    --------------
    tau_hcd: n-array
        HCDs contribution to optical depth
    tau_lya: n-array
        Lyman alpha contribution to optical depth
    tau_tot: n-array
        Total optical depth

    Returns:
    --------------
    Fmean_hcd, Fmean_lya, Fmean_tot: values
        Mean flux value of hcd, lya and total flux
    C: value
        Correlation coefficent between the hcd and the lya fields
    delta_hcd, delta_lya, delta_tot : n-arrays
        Contributions of hcd (flux_hcd) and lya (flux_lya) to the total flux (flux_tot)

    """
    
    # Fluxes
    F_hcd = np.exp(-tau_hcd)
    F_lya = np.exp(-tau_lya)
    F_tot = np.exp(-tau_tot)

    # Mean values and C 
    Fmean_hcd = np.mean(F_hcd)
    Fmean_lya = np.mean(F_lya)
    Fmean_tot = np.mean(F_tot)
    C = Fmean_tot/(Fmean_hcd*Fmean_lya) - 1

    # Deltas
    delta_hcd = F_hcd/Fmean_hcd - 1
    delta_lya = F_lya/Fmean_lya - 1
    delta_tot = F_tot/Fmean_tot - 1

    return Fmean_hcd, Fmean_lya, Fmean_tot, C, delta_hcd, delta_lya, delta_tot


def main(args):

    # Reading the files
        
    with h5py.File(args.data_on, 'r') as f:
        print('Keys:', f.keys())
        header_on = f['Header']
        print('------ Header ------')
        for attr in header_on.attrs:           
            print(f"{attr} : {header_on.attrs[attr]}")
        print('------ Data ------')
        colden = f['colden/H/1'][:]
        print('colden shape:', colden.shape)
        tau_on = f['tau/H/1/1215'][:]
        print('tau shape:', tau_on.shape)

    with h5py.File(args.data_off,'r') as f:
        tau_off = f['tau/H/1/1215'][:]
        print('tau shape:', tau_off.shape)
        
    colden = colden.astype(np.float32)
    tau_on = tau_on.astype(np.float32)
    tau_off = tau_off.astype(np.float32)

    print('----- Simulation box information -----')
    Lbox = 250  # Mpc/h
    print('box size:', Lbox, 'Mpc/h')

    # Number of skewers per side
    Nsk = int(np.sqrt(colden.shape[0]))  # colden.on_shape[0] gives the size of the axis
    print(Nsk,'skewers per side')

    # Number of pixels per skewer
    Np = colden.shape[1] # colden.on_shape[1] gives the size of the columns
    print(Np, 'pixels per skewer')

    # Pixel width 
    Pw = Lbox/Np  # Mpc/h 
    print(Pw, 'Mpc/h pixel width')
    # We are dividing the total box width in comoving units by the number of pixels in each skewer

    # Minimum separation between skewers
    Ssk = Lbox/Nsk  # Mpc/h 
    print(Ssk, 'Mpc/h skewer separation')
    # We are dividing the total box width in comoving units by the number of skewers per side

    # Dividing into miniboxes
    print('-------------- Dividing into %.0f miniboxes -------------' %args.Nmbox)
    (colden_mb, tau_on_mb, tau_off_mb), Nsk_mb = div_box(args.Nmbox, Nsk, Np, colden, tau_on, tau_off)
    print('New data shape:', colden_mb.shape)
    del colden, tau_on, tau_off
    
    print('Minibox information:')
    Lmbox = Lbox/np.sqrt(args.Nmbox)  # Mpc/h
    print('minibox size:', Lmbox, 'Mpc/h')

    # Number of skewers per side
    print(Nsk_mb,'skewers per side of minibox')

    # Number of pixels per skewer
    Np = colden_mb.shape[2]
    print(Np, 'pixels per skewer')

    # Pixel width 
    Pw = Lbox/Np  # Mpc/h 
    print(Pw, 'Mpc/h pixel width')
    
    # Minimum separation between skewers
    Ssk_mb = Lmbox/Nsk_mb  # Mpc/h 
    print(Ssk_mb, 'Mpc/h skewer separation')


    # Iterating over all miniboxes
    masked_los = []
    for mb_index in np.arange(args.Nmbox):
        print('Minibox', mb_index, '...')  
        colden, tau_on, tau_off = colden_mb[mb_index], tau_on_mb[mb_index], tau_off_mb[mb_index]
        # Masking:
        colden_mask, mask_sum, (tau_on_mask, tau_off_mask) = mask_skewers(colden, args.logNHi_min, args.logNHi_max, Nsk_mb, tau_on, tau_off)
        del tau_on, tau_off, colden  # To save memory
        masked_los.append(mask_sum)

        # Different contributions
        tau_hcd, tau_lya, tau_tot = different_contributions(tau_on_mask, tau_off_mask, args.smth_factor, Np, Pw)
        del tau_on_mask, tau_off_mask

        # Calculating deltas
        Fmean_hcd, Fmean_lya, Fmean_tot, C, delta_hcd, delta_lya, delta_tot = deltas(tau_hcd, tau_lya, tau_tot)

        # Saving
        with h5py.File(f"{args.output_dir}/minibox_{mb_index:02d}.hdf5", "w") as f:
        
            f.attrs['logNHI_min'] = args.logNHi_min
            f.attrs['logNHI_max'] = args.logNHi_max
            f.attrs['Smoothing factor'] = args.smth_factor
            
            f.attrs['box_size_Mpch'] = Lbox
            f.attrs['minibox_size_Mpch'] = Lmbox
            f.attrs['skewers_per_side'] = Nsk_mb
            f.attrs['pixels_per_skewer'] = Np
            f.attrs['pixel_width_Mpch'] = Pw
            f.attrs['skewer_separation_Mpch'] = Ssk_mb
    
            grp1 = f.create_group('deltas')
            grp1.create_dataset('delta_hcd', data=delta_hcd)
            grp1.create_dataset('delta_lya', data=delta_lya)
            grp1.create_dataset('delta_tot', data=delta_tot)
    
            grp2 = f.create_group('mean_fluxes')
            grp2.create_dataset('mean_flux_hcd', data=Fmean_hcd)
            grp2.create_dataset('mean_flux_lya', data=Fmean_lya)
            grp2.create_dataset('mean_flux_tot', data=Fmean_tot)

            f.create_dataset('C', data=C)
            f.create_dataset('Colden', data=colden_mask)
            
        print(f'Done and saved in {args.output_dir}/minibox_{mb_index:02d}.hdf5')
    
    masked_los = np.array(masked_los)
    print('Total number of masked los:', Nsk*Nsk-masked_los.sum(), '(', (Nsk*Nsk-masked_los.sum())*100/(Nsk*Nsk), '%)')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process skewer optical depth data")

    parser.add_argument("--data_on", type=str, required=True,
                        help="HDF5 file with damping wings")
    parser.add_argument("--data_off", type=str, required=True,
                        help="HDF5 file without damping wings")
    parser.add_argument("--Nmbox", type=int, default=1,
                        help="Number of miniboxes to split the simulation in")
    parser.add_argument("--logNHi_min", type=float, default=0,
                        help="Minimum log10 column density")
    parser.add_argument("--logNHi_max", type=float, required=True,
                        help="Maximum log10 column density")
    parser.add_argument("--smth_factor", type=float, required=True,
                        help="Smoothing factor")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to where the results will be stored (HDF5 file)")

    args = parser.parse_args()

    main(args)
    
    
    