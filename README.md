# astrid_hcd

Set of code and notebooks to study the contamination from High Column Density absobers (HCDs) to the Lyman alpha forest in the Astrid simulation. 

The scripts assume that there is a folder called `data/` with the following files (provided by Mahdi Qezlou and Simeon Bird):

    - data/spectra_ASTRID_self-shield_off_z2.5_500x500x2500.hdf5
    - data/spectra_ASTRID_z2.5_500x500x2500.hdf5

These files are at NERSC, under `/global/cfs/cdirs/desicollab/users/font/astrid_hcd/data/`

### Possible short projects / tasks

 - study the clustering of absorbers (in real space, no RSD), as a function of column density, and measure b(N_HI) from linear scales
 - measure P1D and PX from the skewers, with and without contamination
 - figure out how to isolate the different components of the model, eq. (4.5) of Font-Ribera & Miralda-Escudé (2012) (https://arxiv.org/abs/1205.2018)
