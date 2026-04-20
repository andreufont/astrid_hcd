import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
import argparse

def main(args):
    Nmbox = len(glob.glob(args.folder_ffts + '/*.hdf5'))
    print(Nmbox, 'Miniboxes found')

    p1d_tot, p1d_lya, p1d_hcd = [], [], []
    p1d_lyahcd = []
    p1d_3lya, p1d_3hcd, p1d_4 = [], [], []
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

        # Calculating P1d contributions and mean values
        # Total
        p1ds_tot = fft_tot*fft_tot.conjugate()*(Lbox/Np**2)  # Same units as Lbox
        p1d_tot.append(np.mean(p1ds_tot, axis=0))

        # Lya
        p1ds_lya = fft_lya*fft_lya.conjugate()*(Lbox/Np**2)  # Same units as Lbox
        p1d_lya.append(np.mean(p1ds_lya, axis=0))

        # Hcd
        p1ds_hcd = fft_hcd*fft_hcd.conjugate()*(Lbox/Np**2)  # Same units as Lbox
        p1d_hcd.append(np.mean(p1ds_hcd, axis=0))

        # LyaxHcd
        p1ds_lyahcd = fft_lya*fft_hcd.conjugate()*Lbox/(Np**2)  # Same units as Lbox
        p1d_lyahcd.append(np.mean(p1ds_lyahcd, axis=0))

        # 3Lya
        p1ds_3lya = fft_lyahcd*fft_lya.conjugate()*Lbox/(Np**2)  # Same units as Lbox
        p1d_3lya.append(np.mean(p1ds_3lya, axis=0))

        # 3Hcd
        p1ds_3hcd = fft_lyahcd*fft_hcd.conjugate()*Lbox/(Np**2)  # Same units as Lbox
        p1d_3hcd.append(np.mean(p1ds_3hcd, axis=0))

        # 4
        p1ds_4 = fft_lyahcd*fft_lyahcd.conjugate()*Lbox/(Np**2) - C_mb**2  # Same units as Lbox
        p1d_4.append(np.mean(p1ds_4, axis=0))

        mb_index += 1

    p1d_tot, p1d_lya, p1d_hcd = np.array(p1d_tot), np.array(p1d_lya), np.array(p1d_hcd)
    p1d_lyahcd = np.array(p1d_lyahcd)
    p1d_3lya, p1d_3hcd, p1d_4 = np.array(p1d_3lya), np.array(p1d_3hcd), np.array(p1d_4)
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

        for m in f.attrs.keys():
            print(f'{m} = {f.attrs[m]}')
            
        f.create_dataset('C', data=C)
        f.create_dataset('k_los', data=k_los)
        f.create_dataset('p1d_tot', data=p1d_tot)
        f.create_dataset('p1d_lya', data=p1d_lya)
        f.create_dataset('p1d_hcd', data=p1d_hcd)
        f.create_dataset('p1d_lyahcd', data=p1d_lyahcd)
        f.create_dataset('p1d_3lya', data=p1d_3lya)
        f.create_dataset('p1d_3hcd', data=p1d_3hcd)
        f.create_dataset('p1d_4', data=p1d_4)

    print(f'Results saved in {args.output_file}.hdf5')

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description= "Compute the different contributions to P1D given the ffts of the miniboxes the simulation has been divided into")

    parser.add_argument("--folder_ffts", type=str, required=True,
                        help="Folder witht the HDF5 files with ffts of the different fields for each minibox")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to HDF5 file where the results will be stored")

    args = parser.parse_args()

    main(args)
