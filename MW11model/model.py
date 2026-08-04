import scipy.constants as scipy_constants
from picca import constants
import numpy as np
from lace.cosmo import cosmology
from scipy.interpolate import RegularGridInterpolator
from forestflow import pcross

from utils import voigt, fft_profile, resample_to_logk



def get_voigt_profile_wave(wave, z, logNHi):
    """Compute voigt profile at input redshift and column density on an observed wavelength grid 
        Parameters,
        ----------
        wave : array
            Observed wavelength (A)
        z : float
            Redshift
        logNHi : float
            Log10 column density (10**N_hi in cm^-2)
    
        Returns
        -------
        array
            Flux corresponding to a voigt profile
    """

    e = 1.6021e-19  # C
    epsilon0 = 8.8541e-12  # C^2.s^2.kg^-1.m^-3
    f = 0.4164
    mp = 1.6726e-27  # kg
    me = 9.109e-31  # kg
    c = scipy_constants.speed_of_light  # m.s^-1
    k = 1.3806e-23  # m^2.kg.s^-2.K-1
    T = 1e4  # K
    gamma = 6.265e8  # s^-1 
    
    lambda_lya = constants.ABSORBER_IGM["LYA"]  # This is the 1215.5 A
        
    wave_rf = wave/(1 + z)  # So we take observed wavelenght and convert it to restframe wavelenght
    
    Deltat_wave = lambda_lya*np.sqrt(2*k*T/mp)/c # Ä Wavelength broadening due to doppler effect 
    
    a = gamma/(4*np.pi*Deltat_wave) * (lambda_lya**2)*(10**(-10))/c # Relation between natural (lorentzian) broadening and thermal (doppler) broadening. See notes for more details 
    u = (wave_rf - lambda_lya)/Deltat_wave  # How "far" are we from the central absorption wavelength normalized with the thermal broadening
    H = voigt(u, np.sqrt(1/2), a) # Voigt profile in Ä taking into account doppler and natural broadening, with a standard deviation of sqrt(1/2) (typical value)

    absorption = H * (e**2) * f * (lambda_lya**2) * 10**(-10) * np.sqrt(np.pi)  # Check notes for detailed explanation, but this is like the effective cross-section of the absorption
    absorption /= (4 * np.pi * epsilon0 * me * (c**2) * Deltat_wave)
    
    # 10^N_hi in cm^-2 and absorb in m^2
    tau = absorption * (10**logNHi) * (10**4)
    return np.exp(-tau)


def profile_wave_to_comov_dist(wave, profile_wave, omegam):

    lambda_lya = constants.ABSORBER_IGM["LYA"]  # This is the 1215.5 A
    r_cosmo = constants.Cosmo(Om=omegam)

    z_value = wave/lambda_lya - 1
    r_comov = r_cosmo.get_r_comov(z_value)  # Mpc/h

    r_linspace = np.linspace(r_comov.min(), r_comov.max(), len(r_comov))  # We need this to be linearly spaced for the fft
    profile_lin = np.interp(r_linspace, r_comov, profile_wave)

    return r_linspace, profile_lin

  
def wave_to_fft_profile(wave, z, logNHi, omegam):
    """Computes Fourier transform of a given wave (observed wavelenght) at input redshift and logNHi.
    
    Steps
    --------
    1. Calls the function 'get_voigt_profile_wave(wave, z, logNHi)' to compute the Voigt profile of the given wave
    2. Calls the function 'profile_wave_to_comov_dist(wave, profile_wavelength, omegam, hubble)' to change from wavelenght to comoving units
    3. Calls the function 'fft_profile(profile)' to perform the Fourier transform of the profile (in comoving units)
    
    
    Parameters
    ----------
    wave : array
        Observed wavelength (A)
    z : float
        Redshift
    logNHi : float
        Log10 column density (10**N_hi in cm^-2)
    omegam : float
        Matter density
    
    Returns
    -------
    (array, array, float)
        (wavenumber grid, voigt profile in Fourier space, dx)"""


    profile_wavelength = get_voigt_profile_wave(wave, z, logNHi)
    profile_wavelength /= np.mean(profile_wavelength) 
    lin_spaced_cmv, profile_cmv = profile_wave_to_comov_dist(wave, profile_wavelength, omegam)
    Deltax = lin_spaced_cmv[1]-lin_spaced_cmv[0]
    k, fft = fft_profile(1-profile_cmv, Deltax)
    
    return k, fft, Deltax



def compute_w1_w2(wave, NHirange, fNHiX, z, hubble, omegam, resample=True, Mpc=True):
    """ Function to compute w1 and w2 following MW11 paper
    
    Parameters
    ------------
    wave : array
        Wavelenght values [A] to compute the Voigt profile
    NHirange : array
        log10(Column density [cm^-2]) array
    fNHIX : array
        f(NHi, X) -column density distribution function per absorption lenght- evaluated at NHirange
    z : float
        Redshift
    hubble : float
        Little hubble constant h
    omegam : float
        omega matter density
    resample : bool
        Wether the user wants the outputs resampled into logarithmic spaced k
    Mpc : bool
        Wether the user wants the outputs in units of h or not. Default is not.
    
    Returns
    ---------
    (array, array, array)
        (k_los [Mpc^-1], w1 [Mpc], w2 [Unitless]) if Mpc = True
        (k_los [h/Mpc], w1 [Mpc/h], w2 [Unitless]) if Mpc = False
    
    """
 
    # Conversion factor dXdComov:
    c_kms = scipy_constants.speed_of_light*1e-3  # km/s
    conversion_factor = 100*((1+z)**2)/c_kms  # h/Mpc 
    cddf = conversion_factor*fNHiX  # h/(Mpc cm^2)

    # Computation of w1 and w2
    w1_int, w2_int = [], []
    for i, nhi_value in enumerate(NHirange):
        k_w1w2, d, Deltax = wave_to_fft_profile(wave, z, nhi_value, omegam=omegam)  # k is in h/Mpc and d in Mpc/h
        w2_int.append(cddf[i]*d*np.log(10)*(10**nhi_value))   # unitless
        w1_int.append(cddf[i]*(d**2)*np.log(10)*(10**nhi_value))  # [Mpc/h]

    w1_int, w2_int = np.array(w1_int), np.array(w2_int)
    w1 = np.trapezoid(w1_int, NHirange, axis=0)
    w2 = np.trapezoid(w2_int, NHirange, axis=0)

    if Mpc:
        k_w1w2 *= hubble  # [Mpc^-1]
        w1 /= hubble  # [Mpc]

    if resample:
        k_los, w1 = resample_to_logk(k_w1w2, w1, k_max=np.pi/(Deltax))
        k_los, w2 = resample_to_logk(k_w1w2, w2, k_max=np.pi/(Deltax))

        return k_los, w1, w2
    
    else:
        return k_w1w2, w1, w2


def _p3d_internal(bias_hcd, bias_lya, z, cosmo, NHirange, fNHiX, kmin=-2.5, kmax=0.5, n_mu=10, Mpc=True):
    """ Function to compute P3D contributions following Mcquinn and White 2011 model -internal version. Same parameters as p3d_model  """

    wave_log = np.linspace(np.log10(2800), np.log10(6800), 2**12) # Angstrom
    wave = 10**wave_log

    # Computing w1 and w2
    k_w1w2, w1, w2 = compute_w1_w2(wave, NHirange, fNHiX, z, cosmo.CAMBparams.h, cosmo.CAMBparams.omegam, resample=True, Mpc=Mpc)
    bias_hcd_flux = 2*w2[1]  # Warning, this needs to be corrected by mean HCD flux absorption

    # Creating (k, mu) grid
    mu = np.linspace(0, 1, n_mu)
    k = np.logspace(kmin, kmax, len(k_w1w2))
    k_los = k*mu[:, None]  # grid
    k_perp = np.sqrt(k**2 - k_los**2)  # grid

    w1_grid = np.interp(k_los, k_w1w2, w1)  # grid
    w2_grid = np.interp(k_los, k_w1w2, w2)  # grid


    # Power spectrum computations
    plin = cosmo.get_linP_Mpc(z, k)
    if Mpc==False:
        plin *= (cosmo.CAMBparams.h)**3  # (Mpc/h)^3
    
    p3d_hcd = (w2_grid**2)*((bias_hcd+mu[:, None]**2)**2)*plin
    
    dkms_dMpc = cosmo.get_dkms_dMpc(z)
    k_d = 0.08*dkms_dMpc  # Mpc^-1
    if Mpc==False:
        k_d /= cosmo.CAMBparams.h  # h/Mpc
    exp = np.exp(-(k_los**2/k_d**2))
    p3d_lya = (bias_lya**2)*((1+mu[:, None]**2)**2)*exp*plin

    p3d_lyahcd = w2_grid*(np.abs(bias_hcd) + mu[:, None]**2)*np.abs(bias_lya)*(1 + mu[:, None]**2)*plin*np.sqrt(exp)

    p3d_tot = p3d_hcd + p3d_lya + 2*p3d_lyahcd

    # Results
    results = {'k':k, 'mu':mu, 'p3d_hcd':p3d_hcd, 'p3d_lya':p3d_lya, 'p3d_lyahcd':p3d_lyahcd,
               'p3d_tot':p3d_tot, 'bias_hcd_flux':bias_hcd_flux, 
               'w1':w1, 'w2':w2, 'k_w1w2':k_w1w2, 'k_perp':k_perp}

    return results

def p3d_model(bias_hcd, bias_lya, z, cosmo, NHirange, fNHiX, kmin=-2.5, kmax=0.5, n_mu=10, Mpc=True):
    """ Function to compute P3D contributions following Mcquinn and White 2011 model

    Parameters:
    ----------------
    bias_hcd: float (>0)
        Large-scale HCD bias ("position bias")
    bias_lya: float (<0)
        Lya-only flux bias
    z : float
        Redshift
    cosmo: LaCE class
        Cosmology
    NHirange: array
        (log10) Column density values where fNHiX has been evaluated
    fNHiX: array
        Column density distribution per absorption length, evaluated at NHirange
    kmin: float, optional
        Minimum value of output k [1/Mpc if Mpc=True, h/Mpc if Mpc=False]
    kmax: float, optional
        Maximum value of output k [1/Mpc if Mpc=True, h/Mpc if Mpc=False]
    n_mu: int, optional
        Number of mu bins
    Mpc: boolean, optional
        If True, the units will be in Mpc (or 1/Mpc). If False, output units in Mpc/h (or h/Mpc)


    Returns:
    ----------------
    (dict)
        [k, mu, p3d_hcd, p3d_lya, p3d_lyaxhcd, p3d_total, bias_hcd_flux]

    """
    internal = _p3d_internal(bias_hcd, bias_lya, z, cosmo, NHirange, fNHiX, kmin=kmin, kmax=kmax, n_mu=n_mu, Mpc=Mpc)

    results = {'k':internal['k'], 'mu':internal['mu'], 'p3d_hcd':internal['p3d_hcd'],
               'p3d_lya':internal['p3d_lya'], 'p3d_lyahcd':internal['p3d_lyahcd'],
               'p3d_tot':internal['p3d_tot'], 'bias_hcd_flux':internal['bias_hcd_flux']}

    return results


def p1d_model(bias_hcd, bias_lya, z, cosmo, NHirange, fNHiX, kmin=-2.5, kmax=0.5, n_mu=10, Mpc=True):
    """ Function to compute P1D contributions following Mcquinn and White 2011 model

    Parameters:
    ----------------
    bias_hcd: float (>0)
        Large-scale HCD bias ("position bias")
    bias_lya: float (<0)
        Lya-only flux bias
    z : float
        Redshift
    cosmo: LaCE class
        Cosmology
    NHirange: array
        (log10) Column density values where fNHiX has been evaluated
    fNHiX: array
        Column density distribution per absorption length, evaluated at NHirange
    kmin: float, optional
        Minimum value of output k [1/Mpc if Mpc=True, h/Mpc if Mpc=False]
    kmax: float, optional
        Maximum value of output k [1/Mpc if Mpc=True, h/Mpc if Mpc=False]
    n_mu: int, optional
        Number of mu bins
    Mpc: boolean, optional
        If True, the units will be in Mpc (or 1/Mpc). If False, output units in Mpc/h (or h/Mpc)


    Returns:
    ----------------
    (dict)
        [klos, p1d_hcd, p1d_lya, p1d_lyaxhcd, p1d_total]

    """
    internal = _p3d_internal(bias_hcd, bias_lya, z, cosmo, NHirange, fNHiX, kmin=kmin, kmax=kmax, n_mu=n_mu, Mpc=Mpc)

    
    # Creating a new grid on KPAR and KPER for integration
    kperp = np.logspace(np.log10(internal['k_perp'][:-1, 0].min()), np.log10(internal['k_perp'][:-1, -1].max()), len(internal['k_perp'][0]))
    KPAR, KPERP = np.meshgrid(internal['k_w1w2'], kperp, indexing='ij')
    K = np.sqrt(KPAR**2 + KPERP**2)
    MU = KPAR/K
    points_target = np.stack([MU.ravel(), K.ravel()], axis=-1)

    # HCD
    interp_hcd = RegularGridInterpolator((internal['mu'], internal['k']), internal['p3d_hcd'], 
                                         bounds_error=False, fill_value=0.0)
    p3d_hcd_newgrid = interp_hcd(points_target).reshape(K.shape)
    p1d_hcd = internal['w1'] + np.trapezoid(kperp*p3d_hcd_newgrid, kperp)/(2*np.pi)

    # LYA
    interp_lya = RegularGridInterpolator((internal['mu'], internal['k']), internal['p3d_lya'], 
                                         bounds_error=False, fill_value=0.0)
    p3d_lya_newgrid = interp_lya(points_target).reshape(K.shape)
    p1d_lya = np.trapezoid(kperp*p3d_lya_newgrid, kperp)/(2*np.pi)

    # LYAxHCD
    interp_lyahcd = RegularGridInterpolator((internal['mu'], internal['k']), internal['p3d_lyahcd'], 
                                         bounds_error=False, fill_value=0.0)
    p3d_lyahcd_newgrid = interp_lyahcd(points_target).reshape(K.shape)
    p1d_lyahcd = np.trapezoid(kperp*p3d_lyahcd_newgrid, kperp)/(2*np.pi)

    # Total
    p1d_tot = p1d_hcd + p1d_lya + 2*p1d_lyahcd

    results={'k_los':internal['k_w1w2'], 'p1d_hcd':p1d_hcd, 'p1d_lya':p1d_lya, 
             'p1d_lyahcd':p1d_lyahcd, 'p1d_tot':p1d_tot}

    return results




def px_model(bias_hcd, bias_lya, z, cosmo, NHirange, fNHiX, r_bins, verbose=True):
    """
    Function to compute px (In Mpc units)

    Parameters:
    ----------------
    bias_hcd: float (>0)
        Large-scale HCD bias ("position bias")
    bias_lya: float (<0)
        Lya-only flux bias
    z : float
        Redshift
    cosmo: LaCE class
        Cosmology
    NHirange: array
        (log10) Column density values where fNHiX has been evaluated
    fNHiX: array
        Column density distribution per absorption length, evaluated at NHirange
    r_bins: array
        Radial bins where the user wants px to be evaluated at. Units must be in acordance to variable Mpc
    

    Returns:
    ----------------
    (dict)
        [k, px_hcd, px_lya, px_lyaxhcd, px_total]

    """
    
    # P3D
    internal = _p3d_internal(bias_hcd, bias_lya, z, cosmo, NHirange, fNHiX, Mpc=True)

    # Functions to pass to pcross

    def _P3D_hcd_px(z, k, mu, p3d_params, new_cosmo_params):

        cosmo_eval = cosmology.Cosmology(cosmo_params_dict=p3d_params)
        plin = cosmo_eval.get_linP_Mpc(z, k)

        kpar = k * mu
        w2_eval = np.interp(kpar, internal['k_w1w2'], internal['w2'])

        p3d_hcd = (w2_eval**2)*((bias_hcd+mu**2)**2)*plin
    
        return p3d_hcd


    def _P3D_lya_px(z, k, mu, p3d_params, new_cosmo_params):

        cosmo_eval = cosmology.Cosmology(cosmo_params_dict=p3d_params)
        dkms_dMpc = cosmo_eval.get_dkms_dMpc(z)
        k_d = 0.08*dkms_dMpc  # Mpc^-1
        plin = cosmo_eval.get_linP_Mpc(z, k)

        kpar = k * mu
        exp = np.exp(-(kpar**2/k_d**2))
        p3d_lya = (bias_lya**2)*((1+mu**2)**2)*exp*plin

        return p3d_lya


    def _P3D_lyahcd_px(z, k, mu, p3d_params, new_cosmo_params):

        cosmo_eval = cosmology.Cosmology(cosmo_params_dict=p3d_params)
        dkms_dMpc = cosmo_eval.get_dkms_dMpc(z)
        k_d = 0.08*dkms_dMpc  # Mpc^-1
        plin = cosmo_eval.get_linP_Mpc(z, k)

        kpar = k * mu
        w2_eval = np.interp(kpar, internal['k_w1w2'], internal['w2'])
        exp = np.exp(-(kpar**2/k_d**2))
    
        p3d_lyahcd = w2_eval*(np.abs(bias_hcd) + mu**2)*np.abs(bias_lya)*(1 + mu**2)*plin*np.sqrt(exp)

        return p3d_lyahcd

    # Cosmo dict
    cosmo_dict = {'H0': 100*cosmo.CAMBparams.h, 
                      'ombh2': cosmo.CAMBparams.ombh2, 'omch2': cosmo.CAMBparams.omch2, 
                      'ns': cosmo.CAMBparams.InitPower.ns, 'As': cosmo.CAMBparams.InitPower.As}
    # HCD
    _P3D_hcd_px.coordinates = 'k_mu'
    px_hcd = []
    for i in np.arange(len(r_bins)-1):  
        r_effs = np.linspace(r_bins[i], r_bins[i+1], 10)
        #print(r_bins[i], '-', r_bins[i+1], 'Mpc')
        px_hcd_eff = pcross.Px_Mpc_detailed([z], internal['k_w1w2'], r_effs, _P3D_hcd_px, 
                                            max_kperp=10.0**2,
                                            p3d_params=cosmo_dict, new_cosmo_params=[])
        px_hcd.append(np.average(px_hcd_eff, axis=1, weights=r_effs))
    px_hcd = np.array(px_hcd)
    if verbose:
        print('HCD contribution computed')

    # Lya
    _P3D_lya_px.coordinates = 'k_mu'
    px_lya = []
    for i in np.arange(len(r_bins)-1):  
        r_effs = np.linspace(r_bins[i], r_bins[i+1], 10)
        px_lya_eff = pcross.Px_Mpc_detailed([z], internal['k_w1w2'], r_effs, _P3D_lya_px, 
                                            max_kperp=10.0**2,
                                            p3d_params=cosmo_dict, new_cosmo_params=[])
        px_lya.append(np.average(px_lya_eff, axis=1, weights=r_effs))
    px_lya = np.array(px_lya)
    if verbose:
        print('Lya contribution computed')


    # LyaxHCD
    _P3D_lyahcd_px.coordinates = 'k_mu'
    px_lyahcd = []
    for i in np.arange(len(r_bins)-1):  
        r_effs = np.linspace(r_bins[i], r_bins[i+1], 10)
        px_lyahcd_eff = pcross.Px_Mpc_detailed([z], internal['k_w1w2'], r_effs, _P3D_lyahcd_px,
                                               max_kperp=10.0**2,
                                               p3d_params=cosmo_dict, new_cosmo_params=[])
        px_lyahcd.append(np.average(px_lyahcd_eff, axis=1, weights=r_effs))
    px_lyahcd = np.array(px_lyahcd)
    if verbose:
        print('LyaxHCD contribution computed')

    px_tot = px_lya + px_hcd + 2*px_lyahcd

    results = {'k_los':internal['k_w1w2'], 'px_lya':px_lya, 'px_hcd':px_hcd, 'px_lyahcd':px_lyahcd, 'px_tot':px_tot}
    
    return results    

    

    

    
    