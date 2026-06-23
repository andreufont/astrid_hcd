import numpy as np


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