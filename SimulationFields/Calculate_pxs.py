import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
import argparse
<<<<<<< HEAD
from Calculate_deltas import mask_skewers
=======

def px_calc(r_bins, row_in, col_in, Nsk, colden_grid, fft_A, fft_B, prt=True):
    if prt:
        print('rbin | # valid pairs | # masked pairs ')
    masked = 0
    jstep = 1
    px_tot = []
    for s in np.arange(len(r_bins)-1):
        px = []
        rows_in, cols_in = row_in[s], col_in[s]
        #if s > 4:
            #jstep = 2
        #if s > 6:
            #jstep = 2
        for i in np.arange(0, Nsk-1, jstep):
            for j in np.arange(0, Nsk-1, jstep):
                i_pairs, j_pairs = (rows_in + i)%Nsk, (cols_in + j)%Nsk
                if (np.isnan(colden_grid[i, j]).sum()) or (np.isnan(colden_grid[i_pairs, j_pairs]).sum()):
                    masked += 1
                    continue
                else:
                    A = fft_A[i, j]
                    B = fft_B[i_pairs, j_pairs].conjugate()
                    px.append(np.real(A*B))
        
        if prt:
            print(s, ' | ', len(px), ' | ', masked)            
        px = np.array(np.mean(px, axis=0))
        px_tot.append(np.mean(px, axis=0))
        
    return np.array(px_tot)  
>>>>>>> wip_px

def main(args):
    Nmbox = len(glob.glob(args.folder_ffts + '/*.hdf5'))
    print(Nmbox, 'Miniboxes found')

<<<<<<< HEAD
    filename = glob.glob(args.folder_ffts + '/*.hdf5')[0]
    with h5py.File(filename, 'r') as f:
        k_los = f['k_los'][:]
        

    px_tot = np.zeros(shape=(Nmbox, args.num_rbins+1, len(k_los)))
    px_lya, px_hcd, px_lyahcd = np.zeros_like(px_tot), np.zeros_like(px_tot), np.zeros_like(px_tot)
    px_3lya, px_3hcd, px_4 = np.zeros_like(px_tot), np.zeros_like(px_tot), np.zeros_like(px_tot)
    C = []
    
=======
    px_tot = []
    px_lya, px_hcd, px_lyahcd = [], [], []
    px_3lya, px_3hcd, px_4 = [], [], []
    C = []

>>>>>>> wip_px
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
<<<<<<< HEAD
            fft_tot = f['fft_tot'][:]
            fft_lya = f['fft_lya'][:]
            fft_hcd = f['fft_hcd'][:]
            fft_lyahcd = f['fft_lyahcd'][:]
            k_los = f['k_los'][:]
=======
            fft_tot = f['fft_tot'][:, :args.index_max]
            fft_lya = f['fft_lya'][:, :args.index_max]
            fft_hcd = f['fft_hcd'][:, :args.index_max]
            fft_lyahcd = f['fft_lyahcd'][:, :args.index_max]
            k_los = f['k_los'][:args.index_max]
            if mb_index == 0:
                print('klos from', k_los[0], 'to', k_los[-1], 'Mpc^-1')
>>>>>>> wip_px
            C_mb = f['C'][()]
            C.append(C_mb)
            colden = f['colden'][:]

<<<<<<< HEAD
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
=======
        # Reshaping fft arrays:
        fft_tot_grid = fft_tot.reshape(Nsk, Nsk, len(k_los))
        fft_lya_grid = fft_lya.reshape(Nsk, Nsk, len(k_los))
        fft_hcd_grid = fft_hcd.reshape(Nsk, Nsk, len(k_los))
        fft_lyahcd_grid = fft_lyahcd.reshape(Nsk, Nsk, len(k_los))

        # Masking
        colden_mask = (colden > 10**args.logNHi_min) & (colden < 10**args.logNHi_max)
        colden_grid = np.where(colden_mask, colden, np.nan).reshape(Nsk, Nsk, Np)

        # Computing radial distances (oly for the first box)
        if mb_index == 0:
            ix = np.linspace(0, Lmbox, Nsk)  # Same units as Lmbox
            iy = ix  # Mpc
            x_grid, y_grid = np.meshgrid(ix, iy, indexing='ij')
            position_grid = np.stack((x_grid, y_grid), axis=-1)  # The -1 is to have shape (Nsk, Nsk, 2) instead of (2, Nsk, Nsk)
            positions = position_grid.reshape(-1, 2)  # x and y positions for each pixel (Same units as Lmbox)
            distances = np.linalg.norm(positions - [0, 0], axis=1)  # These are the distances from the very first pixel to the rest of them (Same units as Lmbox)
            dmin = distances[distances>0].min()  # This is the distance between two consecutive skewers
            r_bins = np.logspace(np.log10(dmin-0.1), np.log10(8), 6)
            print('Radial edges:',  r_bins, '[Lmbox units]')

        # Computing offsets (only for the first box)
            row, col = np.indices((Nsk, Nsk))
            row_in, col_in = [], []
            for s in np.arange(len(r_bins)-1):
                r_mask = (distances >= r_bins[s]) & (distances < r_bins[s+1])
                row_in.append(row.flatten()[r_mask])
                col_in.append(col.flatten()[r_mask])       


        # Calculating Px contributions
        # Total
        fft_A = fft_tot_grid
        fft_B = fft_tot_grid
        px_tot.append(px_calc(r_bins, row_in, col_in, Nsk, colden_grid, fft_A, fft_B)*Lbox/(Np**2))  # Same units as Lbox      

        # Lya
        fft_A = fft_lya_grid
        fft_B = fft_lya_grid
        px_lya.append(px_calc(r_bins, row_in, col_in, Nsk, colden_grid, fft_A, fft_B, prt=False)*Lbox/(Np**2))  # Same units as Lbox

        # Hcd
        fft_A = fft_hcd_grid
        fft_B = fft_hcd_grid
        px_hcd.append(px_calc(r_bins, row_in, col_in, Nsk, colden_grid, fft_A, fft_B, prt=False)*Lbox/(Np**2))  # Same units as Lbox

        # LyaxHcd
        fft_A = fft_lya_grid
        fft_B = fft_hcd_grid
        px_lyahcd.append(px_calc(r_bins, row_in, col_in, Nsk, colden_grid, fft_A, fft_B, prt=False)*Lbox/(Np**2))  # Same units as Lbox

        # 3Lya
        fft_A = fft_lya_grid
        fft_B = fft_lyahcd_grid
        px_3lya.append(px_calc(r_bins, row_in, col_in, Nsk, colden_grid, fft_A, fft_B, prt=False)*Lbox/(Np**2))  # Same units as Lbox

        # 3Hcd
        fft_A = fft_hcd_grid
        fft_B = fft_lyahcd_grid
        px_3hcd.append(px_calc(r_bins, row_in, col_in, Nsk, colden_grid, fft_A, fft_B, prt=False)*Lbox/(Np**2))  # Same units as Lbox

        # 4
        fft_A = fft_lyahcd_grid
        fft_B = fft_lyahcd_grid
        px_4.append(px_calc(r_bins, row_in, col_in, Nsk, colden_grid, fft_A, fft_B, prt=False)*Lbox/(Np**2))  # Same units as Lbox

        mb_index += 1

    px_tot = np.array(px_tot)
    px_lya, px_hcd, px_lyahcd = np.array(px_lya), np.array(px_hcd), np.array(px_lyahcd)
    px_3lya, px_3hcd, px_4 = np.array(px_3lya), np.array(px_3hcd), np.array(px_4)
    C = np.array(C)
    
   # Writing the files out
>>>>>>> wip_px
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

<<<<<<< HEAD
=======

>>>>>>> wip_px
    
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description= "Compute the different contributions to Px given the ffts of the miniboxes the simulation has been divided into")

    parser.add_argument("--folder_ffts", type=str, required=True,
                        help="Folder witht the HDF5 files with ffts of the different fields for each minibox")
<<<<<<< HEAD
    parser.add_argument("--num_rbins", type=int, required=True,
                        help="Number of bins the user wants to divide r into")
    parser.add_argument("--logNHi_min", type=float, default=-10,
=======
    parser.add_argument("--index_max", type=int, required=True,
                        help="Up to what index the user needs to read k_los")
    parser.add_argument("--logNHi_min", type=float, default=0,
>>>>>>> wip_px
                        help="Minimum log10 column density")
    parser.add_argument("--logNHi_max", type=float, required=True,
                        help="Maximum log10 column density")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to HDF5 file where the results will be stored")

    args = parser.parse_args()

    main(args)

        

            

    