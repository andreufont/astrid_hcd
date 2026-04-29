import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
import argparse
from Calculate_deltas import mask_skewers

def main(args):
    Nmbox = len(glob.glob(args.folder_ffts + '/*.hdf5'))
    print(Nmbox, 'Miniboxes found')

    filename = glob.glob(args.folder_ffts + '/*.hdf5')[0]
    with h5py.File(filename, 'r') as f:
        k_los = f['k_los'][:]
        

    px_tot = np.zeros(shape=(Nmbox, args.num_rbins+1, len(k_los)))
    px_lya, px_hcd, px_lyahcd = np.zeros_like(px_tot), np.zeros_like(px_tot), np.zeros_like(px_tot)
    px_3lya, px_3hcd, px_4 = np.zeros_like(px_tot), np.zeros_like(px_tot), np.zeros_like(px_tot)
    C = []
    
    mb_index = 0
    for filename in sorted(glob.glob(args.folder_ffts + '/*.hdf5')):
        print('Minibox', mb_index, '...')

        # Reading fft files
        with h5py.File(filename, 'r') as f:
            
            # Atributes
            logNHi_min = f.attrs['logNHI_min']
            logNHi_max = f.attrs['logNHI_max']
            smth_factor = f.attrs['Smoothing factor']
            Lbox = f.attrs['box_size_Mpch']
            Lmbox = f.attrs['minibox_size_Mpch']
            Nsk = f.attrs['skewers_per_side']
            Np = f.attrs['pixels_per_skewer']
            Pw = f.attrs['pixel_width_Mpch']
            Ssk = f.attrs['skewer_separation_Mpch']

            # Data
            fft_tot = f['fft_tot'][:]
            fft_lya = f['fft_lya'][:]
            fft_hcd = f['fft_hcd'][:]
            fft_lyahcd = f['fft_lyahcd'][:]
            k_los = f['k_los'][:]
            C_mb = f['C'][()]
            C.append(C_mb)
            colden = f['colden'][:]

        # Computing r distances
        ix, iy = np.divmod(np.arange(Nsk), Lmbox)  # This gives me the coordinates of each skewer within the minibox
        dx = ix[:, None] - ix[None, :]
        dy = iy[:, None] - iy[None, :]
        dr = np.sqrt(dx**2 + dy**2)*Ssk  # This matrix contains the radial distance (Mpc/h) between all skewers within the minibox

        # Setting r bins (the same for all miniboxes)
        if mb_index == 0:
            r_bins = np.linspace(dr.min(), dr.max(), args.num_rbins+1)  # radial bins Mpc/h
            print(len(r_bins), 'radial edges (Mpc/h) defined to have', len(r_bins)-1, 'radial bins:', r_bins, '[Mpc/h]')

        # Masking
        _, _, (fft_tot_mask, fft_lya_mask, fft_hcd_mask, fft_lyahcd_mask, dr_flatten_mask) = mask_skewers(colden, 
                                                                                                          args.logNHi_min, args.logNHi_max, Nsk, 
                                                                                                          fft_tot, fft_lya, fft_hcd, fft_lyahcd, dr.flatten())

        # Computing Px contributions and mean values
        # Total
        for i, r_value in enumerate(r_bins[:-1]):
            r_mask = (dr_flatten_mask > r_value) & (dr_flatten_mask < r_bins[i+1])
            fft_tot_2mask = fft_tot_mask[r_mask]
    
            pxs_tot = fft_tot_2mask*fft_tot_2mask.conjugate()*Lbox/(Np**2)
            px_tot[mb_index, i, :] = np.mean(pxs_tot, axis=0)

        # Lya
        for i, r_value in enumerate(r_bins[:-1]):
            r_mask = (dr_flatten_mask > r_value) & (dr_flatten_mask < r_bins[i+1])
            fft_lya_2mask = fft_lya_mask[r_mask]
    
            pxs_lya = fft_lya_2mask*fft_lya_2mask.conjugate()*Lbox/(Np**2)
            px_lya[mb_index, i, :] = np.mean(pxs_lya, axis=0)

        # HCDs
        for i, r_value in enumerate(r_bins[:-1]):
            r_mask = (dr_flatten_mask > r_value) & (dr_flatten_mask < r_bins[i+1])
            fft_hcd_2mask = fft_hcd_mask[r_mask]
    
            pxs_hcd = fft_hcd_2mask*fft_hcd_2mask.conjugate()*Lbox/(Np**2)
            px_hcd[mb_index, i, :] = np.mean(pxs_hcd, axis=0)

        # Lya x HCDs
        for i, r_value in enumerate(r_bins[:-1]):
            r_mask = (dr_flatten_mask > r_value) & (dr_flatten_mask < r_bins[i+1])
            fft_lya_2mask = fft_lya_mask[r_mask]
            fft_hcd_2mask = fft_hcd_mask[r_mask]
    
            pxs_lyahcd = fft_lya_2mask*fft_hcd_2mask.conjugate()*Lbox/(Np**2)
            px_lyahcd[mb_index, i, :] = np.mean(pxs_lyahcd, axis=0)

        # 3Lya
        for i, r_value in enumerate(r_bins[:-1]):
            r_mask = (dr_flatten_mask > r_value) & (dr_flatten_mask < r_bins[i+1])
            fft_lya_2mask = fft_lya_mask[r_mask]
            fft_lyahcd_2mask = fft_lyahcd_mask[r_mask]
    
            pxs_3lya = fft_lyahcd_2mask*fft_lya_2mask.conjugate()*Lbox/(Np**2)
            px_3lya[mb_index, i, :] = np.mean(pxs_3lya, axis=0)

        # 3HCD
        for i, r_value in enumerate(r_bins[:-1]):
            r_mask = (dr_flatten_mask > r_value) & (dr_flatten_mask < r_bins[i+1])
            fft_hcd_2mask = fft_hcd_mask[r_mask]
            fft_lyahcd_2mask = fft_lyahcd_mask[r_mask]
    
            pxs_3hcd = fft_lyahcd_2mask*fft_hcd_2mask.conjugate()*Lbox/(Np**2)
            px_3hcd[mb_index, i, :] = np.mean(pxs_3hcd, axis=0)

        # 4
        for i, r_value in enumerate(r_bins[:-1]):
            r_mask = (dr_flatten_mask > r_value) & (dr_flatten_mask < r_bins[i+1])
            fft_lyahcd_2mask = fft_lyahcd_mask[r_mask]
    
            pxs_4 = fft_lyahcd_2mask*fft_lyahcd_2mask.conjugate()*Lbox/(Np**2)
            px_4[mb_index, i, :] = np.mean(pxs_4, axis=0)
        
        mb_index += 1
        
    C = np.array(C)

    # Writing the files out
    with h5py.File(f'{args.output_file}.hdf5', 'w') as f:
    
        f.attrs['logNHI_min'] = logNHi_min
        f.attrs['logNHI_max'] = logNHi_max
        f.attrs['Smoothing factor'] = smth_factor
            
        f.attrs['minibox_size_Mpch'] = Lbox
        f.attrs['skewers_per_side'] = Nsk
        f.attrs['pixels_per_skewer'] = Np
        f.attrs['pixel_width_Mpch'] = Pw
        f.attrs['skewer_separation_Mpch'] = Ssk
        f.attrs['----- WARNING -----'] = 'Units in terms of h'

        for m in f.attrs.keys():
            print(f'{m} = {f.attrs[m]}')
            
        f.create_dataset('C', data=C)
        f.create_dataset('r_bins', data=r_bins)
        f.create_dataset('k_los', data=k_los)
        f.create_dataset('px_tot', data=px_tot)
        f.create_dataset('px_lya', data=px_lya)
        f.create_dataset('px_hcd', data=px_hcd)
        f.create_dataset('px_lyahcd', data=px_lyahcd)
        f.create_dataset('px_3lya', data=px_3lya)
        f.create_dataset('px_3hcd', data=px_3hcd)
        f.create_dataset('px_4', data=px_4)

    print(f'Results saved in {args.output_file}.hdf5')

    
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description= "Compute the different contributions to Px given the ffts of the miniboxes the simulation has been divided into")

    parser.add_argument("--folder_ffts", type=str, required=True,
                        help="Folder witht the HDF5 files with ffts of the different fields for each minibox")
    parser.add_argument("--num_rbins", type=int, required=True,
                        help="Number of bins the user wants to divide r into")
    parser.add_argument("--logNHi_min", type=float, default=-10,
                        help="Minimum log10 column density")
    parser.add_argument("--logNHi_max", type=float, required=True,
                        help="Maximum log10 column density")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to HDF5 file where the results will be stored")

    args = parser.parse_args()

    main(args)

        

            

    