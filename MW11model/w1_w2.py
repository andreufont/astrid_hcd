import scipy.constants as scipy_constants
import scipy.special
from picca import constants
import numpy as np
import argparse
import h5py

def voigt(x, sigma=1, gamma=1):
    return np.real(scipy.special.wofz((x + 1j*gamma)/(sigma*np.sqrt(2))))

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
    #H = scipy.special.voigt_profile(u, np.sqrt(1/2), a) # Voigt profile in Ä taking into account doppler and natural broadening, with a standard deviation of sqrt(1/2) (typical value)

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
    
    size = profile.size
    ft_profile = dx * np.fft.rfft(profile) # The dx factor is included to account for the discretization of the integration (check notes)
    k = np.fft.rfftfreq(size, dx) * (2 * np.pi) # 2pi factor is to obtain k in h/Mpc^-1 instead of in frecuency values
 
    return k, np.abs(ft_profile)

  
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



def fNHiX_to_fn(fNHiX, NHirange):
    """
    Function to change from f(NHi, X) -column density distribution function per absorption length- to f(n) in Tan25 (normalized)
     
    Parameters
    -----------
    fNHiX : array
        f(NHi, X) function in terms of NHi range [cm^2]
    NHirange : array
        log10 (column density values [cm^-2]) 
    
    Returns
    -----------
    (array)
        f(n) function in terms of NHirange"""
    
    fn = []
    for i in np.arange(len(NHirange)):
        fn.append(fNHiX[i] * 10**(NHirange[i]))
    fn = np.array(fn) * np.log(10)
    fn /=  np.trapezoid(fn, NHirange)

    return fn



def fn_to_fNHi(fn, NHirange, dX, norm=False, norm_factor=None):
    """
    Function to change from f(n) in Tan25 to f(NHi, X) -column density distribution function per absorption length
    
    Parameters
    -------------
    fn : array
        f(n) function from Tan25 [unitless]
    NHirange : array
        log10 (column density values [cm^-2]) the function f(n) is evaluated at
    dX : array
        Total absorption length
    norm : bool
        If True, f(n) input has been normalized to the area.
        If False, f(n) has not been noralized (counts/bin_width)
    norm_factor : float, optional
        Required if norm=True
        Value used to normalized f(n) (integrated area of unnormalized f(n))

    Returns
    ----------
    array
        f(Nhi, X) function
    """

    if norm and norm_factor is None:
        raise ValueError('norm_factor must be provided when norm=True')
    
    if norm:
        fn *= norm_factor
    
    fnX = fn/dX
    fNHi = fnX/(np.log(10)*(10**NHirange))

    return fNHi



def compute_w1_w2(wave, NHirange, fNHiX, z, hubble, omegam, resample=True):
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
    
    Returns
    ---------
    (array, array, array)
        (k_los [Mpc^-1], w1 [Mpc], w2 [Unitless])
    
    """
 
    # Conversion factor dXdComov:
    c_kms = scipy_constants.speed_of_light*1e-3  # km/s
    conversion_factor = 100*hubble*((1+z)**2)/c_kms
    cddf = conversion_factor*fNHiX

    # Computation of w1 and w2
    w1_int, w2_int = [], []
    for i, nhi_value in enumerate(NHirange):
        k_w1w2, d, Deltax = wave_to_fft_profile(wave, z, nhi_value, omegam=omegam)  # k is in h/Mpc and d in Mpc/h
        k_w1w2 /= hubble  # [Mpc^-1]
        d *= hubble  # [Mpc]
        w2_int.append(cddf[i]*d*np.log(10)*(10**nhi_value))
        w1_int.append(cddf[i]*(d**2)*np.log(10)*(10**nhi_value))

    w1_int, w2_int = np.array(w1_int), np.array(w2_int)
    w1 = np.trapezoid(w1_int, NHirange, axis=0)
    w2 = np.trapezoid(w2_int, NHirange, axis=0)

    if resample:
        k_los, w1 = resample_to_logk(k_w1w2, w1, k_max=np.pi/(Deltax*2))
        k_los, w2 = resample_to_logk(k_w1w2, w2, k_max=np.pi/(Deltax*2))

        return k_los, w1, w2

        
    else:
        return k_w1w2, w1, w2
    

