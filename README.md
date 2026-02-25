# astrid_hcd

Set of code and notebooks to study the contamination from High Column Density absobers (HCDs) to the Lyman alpha forest in the Astrid simulation. 

The scripts assume that there is a folder called `data/` with the following files (provided by Mahdi Qezlou and Simeon Bird):

    - data/spectra_ASTRID_self-shield_off_z2.5_500x500x2500.hdf5
    - data/spectra_ASTRID_z2.5_500x500x2500.hdf5

These files are at NERSC, under `/global/cfs/cdirs/desicollab/users/font/astrid_hcd/data/`

## Folders

 - HCDclustering: Work in progress related to the calculation of P(k) and the correlation function -and their respective bias values- from the simulation box
 - MW11model: Work in progress related to the modelling of HCDs 3D and 1D power spectrum following MW11
 - SimulationFields: Work in progress related to the extraction of different fields (lya, hcd, total), their mean flux, their deltas, etc from the simulation box.

 - (future) astrid_hcd: Final scripts


### Possible short projects / tasks

 - study the clustering of absorbers (in real space, no RSD), as a function of column density, and measure b(N_HI) from linear scales
 - measure P1D and PX from the skewers, with and without contamination
 - figure out how to isolate the different components of the model, eq. (4.5) of Font-Ribera & Miralda-Escudé (2012) (https://arxiv.org/abs/1205.2018)
