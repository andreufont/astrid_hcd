import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
import argparse

def main(args):
    Nmbox = len(glob.glob(args.folder_deltas + '/*.hdf5'))
    print(Nmbox, 'Miniboxes found')

    mb_index = 0
    for filename in sorted(glob.glob(args.folder_deltas + '/*.hdf5')):
        print('Minibox', mb_index, '...')

        # Reading delta files
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
            delta_hcd = f['deltas/delta_hcd'][:]
            delta_lya = f['deltas/delta_lya'][:]
            delta_tot = f['deltas/delta_tot'][:]
            C = f['C'][()] 

            colden = f['Colden'][:]

        # Calculating k_los
        k_los = 2*np.pi*np.fft.rfftfreq(Np, d=Pw)  # In h/Mpc

        # Creating a file to store the ffts:
        with h5py.File(f"{args.output_dir}/minibox_{mb_index:02d}.hdf5", 'w') as f:
    
            f.attrs['logNHI_min'] = logNHi_min
            f.attrs['logNHI_max'] = logNHi_max
            f.attrs['Smoothing factor'] = smth_factor

            f.attrs['box_size_Mpch'] = Lbox
            f.attrs['minibox_size_Mpch'] = Lmbox
            f.attrs['skewers_per_side'] = Nsk
            f.attrs['pixels_per_skewer'] = Np
            f.attrs['pixel_width_Mpch'] = Pw
            f.attrs['skewer_separation_Mpch'] = Ssk
            
            f.create_dataset('C', data=C)
            f.create_dataset('k_los', data=k_los)
            f.create_dataset('colden', data=colden)

        # Computing (and saving) ffts
        fft_tot = np.fft.rfft(delta_tot, axis=-1)
        fft_lya = np.fft.rfft(delta_lya, axis=-1)
        fft_hcd = np.fft.rfft(delta_hcd, axis=-1)
        fft_lyahcd = np.fft.rfft(delta_hcd*delta_lya, axis=-1)

        with h5py.File(f"{args.output_dir}/minibox_{mb_index:02d}.hdf5", 'a') as f:
       
            f.create_dataset('fft_tot', data=fft_tot)
            f.create_dataset('fft_lya', data=fft_lya)
            f.create_dataset('fft_hcd', data=fft_hcd)
            f.create_dataset('fft_lyahcd', data=fft_lyahcd)

        del fft_tot, fft_lya, fft_hcd, fft_lyahcd
        
        print(f'Results saved in {args.output_dir}/minibox_{mb_index:02d}.hdf5')
        mb_index += 1
 
    return
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description= "Compute the different fft fields given the deltas")

    parser.add_argument("--folder_deltas", type=str, required=True,
                        help="Folder witht the HDF5 files with flux deltas of the different field for each minibox")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to folder where the results will be stored (in HDF5 files)")

    args = parser.parse_args()

    main(args)

