import scipy.constants
import scipy.special
import scipy.integrate as integrate
from picca import constants
import numpy as np
from astropy.cosmology import FlatLambdaCDM

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
    c = scipy.constants.speed_of_light  # m.s^-1
    k = 1.3806e-23  # m^2.kg.s^-2.K-1
    T = 1e4  # K
    gamma = 6.265e8  # s^-1 
    
    lambda_lya = constants.ABSORBER_IGM["LYA"]  # This is the 1215.5 A
        
    wave_rf = wave/(1 + z)  # So we take observed wavelenght and convert it to restframe wavelenght
    
    Deltat_wave = lambda_lya*np.sqrt(2*k*T/mp)/c # Ä Wavelength broadening due to doppler effect 
    
    a = gamma/(4*np.pi*Deltat_wave) * (lambda_lya**2)*(10**(-10))/c # Relation between natural (lorentzian) broadening and thermal (doppler) broadening. See notes for more details 
    u = (wave_rf - lambda_lya)/Deltat_wave  # How "far" are we from the central absorption wavelength normalized with the thermal broadening
    H = scipy.special.voigt_profile(u, np.sqrt(1/2), a) # Voigt profile in Ä taking into account doppler and natural broadening, with a standard deviation of sqrt(1/2) (typical value)

    absorption = H * (e**2) * f * (lambda_lya**2) * 10**(-10)  # Check notes for detailed explanation, but this is like the effective cross-section of the absorption
    absorption /= (4  * epsilon0 * me * (c**2) * Deltat_wave)
    
    # 10^N_hi in cm^-2 and absorb in m^2
    tau = absorption * (10**logNHi) * (10**4)
    return np.exp(-tau)


def profile_wave_to_comov_dist(wave, profile_wave, omegam, hubble):

    lambda_lya = constants.ABSORBER_IGM["LYA"]  # This is the 1215.5 A
    c = scipy.constants.speed_of_light  # ms^-1
    r_cosmo = FlatLambdaCDM(H0=hubble*100, Om0=omegam)

    z_value = wave/lambda_lya - 1
    r_comov = np.array(r_cosmo.comoving_distance(z_value))  # Mpc
    lin_spaced_comov_dist = np.linspace(r_comov[0], r_comov[-1], r_comov.size)
    profile_comov_dist_ext = np.interp(lin_spaced_comov_dist, r_comov, profile_wave)

    return lin_spaced_comov_dist, profile_comov_dist_ext

def fft_profile(profile, dx):
    """Compute Fourier transform of a voigt profile

    Parameters
    ----------
    profile : array
        Input voigt profile in real space (function of comoving distance)
    dx : float
        Comoving distance bin size

    Returns
    -------
    (array, array)
        (wavenumber grid, voigt profile in Fourier space)
    """
    # not normalized
    size = profile.size
    ft_profile = dx * np.fft.rfft(1-profile) # The dx factor is included to account for the discretization of the integration (check notes)
    k = np.fft.rfftfreq(size, dx) * (2 * np.pi) # 2pi factor is to obtain k in rad/Mpc/h instead of in frecuency values
 
    return k, np.abs(ft_profile) # We need the abs value because rfft only guarantees >0 results, but they can be imaginary"
  
def wave_to_fft_profile(wave, z, logNHi, omegam, omegal, hubble):
    """Computes Fourier transform of a given wave (observed wavelenght) at input redshift and logNHi.
    
    Steps
    --------
    1. Calls the function 'get_voigt_profile_wave(wave, z, logNHi)' to compute the Voigt profile of the given wave
    2. Calls the function 'profile_wave_to_comov_dist(wave, profile_wave, cosmo, differential=False)' to change from wavelenght to comoving units
    3. Calls the function 'fft_profile(profile)' to perform the Fourier transform of the profile (in comoving units)
    
    
    Parameters
    ----------
    wave : array
    Observed wavelength (A)
    z : float
        Redshift
    logNHi : float
        Log10 column density (10**N_hi in cm^-2)
    cosmo : picca.constants.Cosmo
        Cosmology object.   
    
    Returns
    -------
    (array, array)
        (wavenumber grid, voigt profile in Fourier space)"""


    
    profile_wavelength = get_voigt_profile_wave(wave, z, logNHi)
    profile_wavelength /= np.mean(profile_wavelength) 
    lin_spaced_cmv, profile_cmv = profile_wave_to_comov_dist(wave, profile_wavelength, omegam, hubble)
    Deltax = lin_spaced_cmv[1]-lin_spaced_cmv[0]
    k, fft = fft_profile(profile_cmv, np.abs(Deltax))
    
    return k, fft, Deltax


def resample_to_logk(k, w, k_min=None, k_max=None):
    """
    Resample function to log-spaced k with physically motivated limits.
    
    Parameters
    -------------
    k : array
        Wavenumber linear spaced array
    w : array(s)
        Associated function
    
    Returns
    ----------
    (array, array(s))
        (log-spaced k [h/Mpc], log-spaced function)"""
    
    if k_min is None:
        k_min = k[k > 0].min()
    if k_max is None:
        k_max = k.max()
    
    n_points = 2*len(k)
    k_log = np.logspace(np.log10(k_min), np.log10(k_max), n_points)
    w_log = np.interp(k_log, k, w, left=0, right=0)
    
    return k_log, w_log


    