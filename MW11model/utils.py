import numpy as np
import scipy.special
import scipy.constants
from lace.cosmo import cosmology


def voigt(x, sigma=1, gamma=1):
    return np.real(scipy.special.wofz((x + 1j*gamma)/(sigma*np.sqrt(2))))

def fft_profile(profile, dx):
    """Compute Fourier transform of a profile 

    Parameters
    ----------
    profile : array
        Input profile in real space (function of comoving distance)
    dx : float
        Comoving distance bin size

    Returns
    -------
    (array, array)
        (wavenumber grid, Profile in Fourier space)
    """
    
    size = profile.size
    ft_profile = dx * np.fft.rfft(profile) # The dx factor is included to account for the discretization of the integration (check notes)
    k = np.fft.rfftfreq(size, dx) * (2 * np.pi) # 2pi factor is to obtain k in inverse dx units instead of in frecuency values
 
    return k, np.abs(ft_profile)
    
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

def cddf_to_fNHiX(colden, NHirange, Lbox, Nsk, z):
    """ Function to change from column density distribution of a certain box to f(NHi, X) -column density distribution function per absorption length

    Parameters:
    --------------
    colden: array
        Column density values (log10 [cm^-2])
    NHirange: array
        log10 (column density values [cm^-2]) the user wants the fNHiX to be evaluated at
    Lbox: float
        Box size. WARNING, in Mpc/h
    Nsk: int
        Number of skewers
    z: float
        Redshift


    Returns:
    -------------
    array
        f(Nhi, X) function
    """

    # Computing the cddf
    counts, _ = np.histogram(colden, bins=NHirange)
    mid_nhi = (NHirange[:-1] + (NHirange[1] - NHirange[0])/2)  # Middle value of each bin
    dn = mid_nhi[1]-mid_nhi[0]  # Step

    c_kms = scipy.constants.c*1e-3  # kms^-1
    dX = (Nsk**2)*Lbox*100*((1+z)**2)/c_kms  # Absorption length of the simulation
    # WARNING: Lbox must be in Mpc/h
    
    # fNHiX
    fnX = np.interp(NHirange, mid_nhi, counts/dn)/dX
    fNHiX = fnX/(np.log(10)*(10**NHirange))

    return fNHiX

    
def resample_to_logk(k, w, k_min=None, k_max=None):
    """
    Resample function w to log-spaced k with physically motivated limits.
    
    Parameters
    -------------
    k : array
        Wavenumber linear spaced array
    w : array(s)
        Associated function
    
    Returns
    ----------
    (array, array(s))
        (log-spaced k, log-spaced function)"""
    
    if k_min is None:
        k_min = k[k > 0].min()
    if k_max is None:
        k_max = k.max()
    
    n_points = 2*len(k)
    k_log = np.logspace(np.log10(k_min), np.log10(k_max), n_points)
    w_log = np.interp(k_log, k, w, left=0, right=0)
    
    return k_log, w_log



class Cosmologies:
 
    _PRESETS = {
        'MW11': {'hubble': 0.71, 
                 'omegam': 0.27, 'ombh2': 0.046*(0.71**2), 'omch2': (0.27-0.046)*(0.71**2),
                 'ns': 0.96, 'As': 2e-9},
        'ASTRID': {'hubble': 0.6774, 
                 'omegam': 0.3089, 'ombh2': 0.0486*(0.6774**2), 'omch2': (0.3089-0.0486)*(0.6774**2),
                 'ns': 0.9667, 'As': 2.142e-9},
    }

    
    def get_cosmo(self, name):
        
        p = self._PRESETS[name] 
        cosmo = cosmology.Cosmology()
        cosmo.CAMBparams.set_cosmology(H0=100*p['hubble'], ombh2=p['ombh2'], omch2=p['omch2'])
        cosmo.CAMBparams.InitPower.set_params(As=p['As'], ns=p['ns'])

        return cosmo

    def cosmo_dict(self, name):
        """ Returns dict with cosmo params """
        p = self._PRESETS[name]
        cosmo_dict = {'H0': 100*p['hubble'], 
                      'ombh2': p['ombh2'], 'omch2': p['omch2'], 
                      'ns': p['ns'], 'As': p['As']}
 
        return cosmo_dict