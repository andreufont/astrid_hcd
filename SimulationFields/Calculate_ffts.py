import numpy as np
import matplotlib.pyplot as plt
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

    mb_per_side = int(np.sqrt(Nbox))
    mb_size = Nsk // mb_per_side  # Number of skewers per side inside each mb

    # Computing slices
    slices = []
    for j_mb in range(mb_per_side):  
        for i_mb in range(mb_per_side):
            imin, imax = i_mb*mb_size, (i_mb+1)*mb_size
            jmin, jmax = j_mb*mb_size, (j_mb+1)*mb_size
            slices.append((imin, imax, jmin, jmax))

    results = []
    for arr in arrays:
        grid = arr.reshape(Nsk, Nsk, Np)
        new_array = np.zeros((Nbox, mb_size**2, Np))

        for mb_id, (imin, imax, jmin, jmax) in enumerate(slices):
            minibox = grid[imin:imax, jmin:jmax, :]
            new_array[mb_id] = minibox.reshape(-1, Np)

        results.append(new_array) 

    return results


def main(args):

    # Reading files:
    with h5py.File(args.data_deltas, 'r') as f:
        print('Atributes:')
        for k in f.attrs.keys():
            print(f'{k} = {f.attrs[k]}')
        logNHi_min = f.attrs['logNHI_min']
        logNHi_max = f.attrs['logNHI_max']
        smth_factor = f.attrs['Smoothing factor']
        print('----------------') 
        print('Data:')
        print(f.keys())
        delta_hcd = f['deltas/delta_hcd'][:]
        print('delta_hcd shape:', delta_hcd.shape)
        delta_lya = f['deltas/delta_lya'][:]
        print('delta_lya shape:', delta_lya.shape)
        delta_tot = f['deltas/delta_tot'][:]
        print('delta_tot shape:', delta_tot.shape)
        C = f['C'][()]  # Scalar data
        print('C:', C)

    print('----- Useful information -----')
    Lbox = 250  # Mpc/h
    print('box size:', Lbox, 'Mpc/h')

    # Number of skewers per side
    Nsk = 500  # colden.on_shape[0] gives the size of the axis
    print(Nsk,'skewers per side')

    # Number of pixels per skewer
    Np = 2500 # colden.on_shape[1] gives the size of the columns
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
    print('---------- Dividing into', args.Nbox, 'miniboxes -----------')
    delta_hcd_mb, delta_lya_mb, delta_tot_mb = div_box(args.Nbox, Nsk, Np, delta_hcd, delta_lya, delta_tot)
    del delta_hcd, delta_lya, delta_tot

    # Calculating k_los
    k_los = 2*np.pi*np.fft.rfftfreq(Np, d=Pw)  # In h/Mpc

    # Creating a file to store the ffts:
    with h5py.File(args.output_dir, 'w') as f:
    
        f.attrs['logNHI_min'] = logNHi_min
        f.attrs['logNHI_max'] = logNHi_max
        f.attrs['Smoothing factor'] = smth_factor
        f.attrs['Number of miniboxes'] = args.Nbox
        f.create_dataset('C', data=C)
        f.create_dataset('k_los', data=k_los)

    # Computing (and saving) ffts
    print('----------- Computing FFTs ----------')
    for mb in range(args.Nbox):
        print('Minibox', mb)
        fft_tot_mb = np.fft.rfft(delta_tot_mb[mb])
        fft_lya_mb = np.fft.rfft(delta_lya_mb[mb])
        fft_hcd_mb = np.fft.rfft(delta_hcd_mb[mb])
        fft_lyahcd_mb = np.fft.rfft(delta_hcd_mb[mb]*delta_lya_mb[mb])

        with h5py.File(args.output_dir, 'a') as f:
   
            grp1 = f.create_group('Minibox%.0f' %mb)
            grp1.create_dataset('fft_tot', data=fft_tot_mb)
            grp1.create_dataset('fft_lya', data=fft_lya_mb)
            grp1.create_dataset('fft_hcd', data=fft_hcd_mb)

        del fft_tot_mb, fft_lya_mb, fft_hcd_mb, fft_lyahcd_mb
        
    print('Results saved in', args.output_dir)
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Divide simulation box in Nbox miniboxes and compute the different fft fields")

    parser.add_argument("--data_deltas", type=str, required=True,
                        help="HDF5 file with flux deltas of the different field for the original simulation box")
    parser.add_argument("--Nbox", type=int, required=True,
                        help="Number of miniboxes")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to where the results will be stored (HDF5 file)")

    args = parser.parse_args()

    main(args)

