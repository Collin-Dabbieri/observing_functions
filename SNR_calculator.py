import numpy as np
from scipy.optimize import fsolve


def calculate_SNR(Rs,Rb,Rd,Nr,npix,t):
    '''
    calculates the signal-to-noise ratio
    params:
        Rs - source count rate (photons s^-1)
        Rb - background count rate (photons s^-1 pixel^-1)
        Rd - dark current (electrons s^-1 pixel^-1)
        Nr - read noise (electrons)
        npix - number of pixels for PSF of source
        t - observation time (s)
    returns:
        SNR - signal to noise ratio
    '''

    SNR=(Rs*t)/np.sqrt(Rs*t+npix*(Rb*t+Rd*t+Nr**2))

    return SNR

def calculate_exposure_time(Rs,Rb,Rd,Nr,npix,SNR,guess=500):
    '''
    calculates the exposure time for a given SNR

    params:
       Rs - source count rate (photons s^-1)
       Rb - background count rate (photons s^-1 pixel^-1)
       Rd - dark current (electrons s^-1 pixel^-1)
       Nr - read noise (electrons)
       npix - number of pixels for PSF of source
       SNR - signal to noise ratio
       guess - guess for observation time
    returns:
        t - observation time (s)
    '''

    def func(t):
        return Rs*t/np.sqrt(Rs*t+npix*(Rb*t+Rd*t+Nr**2)) - SNR

    time=fsolve(func,guess)
    return time

def get_source_rate(m,zero_point,WL_eff,width_eff,D,epsilon):
    '''
    Calculates the source rate in photons s^-1 from an apparent magnitude

    params:
        m - apparent magnitude
        zero_point - zero magnitude flux for this band (erg s^-1 cm^-2 Angstrom^-1)
        WL_eff - effective wavelength of this band (Angstrom)
        width_eff - effective width of this band (Angstrom)
        D - Diameter of telescope (meters)
        epsilon - efficiency (telescope and QE)

    returns:
        Rs - source rate (photons s^-1)
    '''

    # Convert apparent mag to flux
    F=epsilon*zero_point*10**(-0.4*m)

    # Find energy of effective photon
    h=6.626e-27 #erg s
    c=3e18 # Angstroms s^-1
    E_photon=h*c/WL_eff

    # find photon flux
    F_photon=F/E_photon #photons s^-1 cm^-2 Angstrom^-1

    # multiply by area of telescope and effective wavelength
    R_cm=100*D/2

    Rs=F_photon*np.pi*R_cm**2*width_eff #photons s^-1

    return Rs

def get_platescale(R,D):
    '''
    Calculates the platescale

    params:
        R - telescope resolution (f=Rd)
        D - telescope diameter (meters)
    returns:
        s - platescale (arcsec micron^-1)
    '''
    f=R*D #meters
    f_microns=f*1000000
    s=206265/f_microns #arcsec micron^-1
    return s

def get_background_rate(mu,zero_point,WL_eff,width_eff,R,D,pixel_length,epsilon):
    '''
    Calculates background rate in photons s^-1 pixel^-1 from surface brightness

    params:
        mu - background surface brightness in mag arcsec^-2
        zero_point - zero magnitude flux for this band (erg s^-1 cm^-2 Angstrom^-1)
        WL_eff - effective wavelength for this band (Angstrom)
        R - telescope resolution (f=RD)
        D - telescope diameter (meters)
        pixel_length - length of a pixel (microns)
        epsilon - efficiency (telescope and QE)

    returns:
        Rb - background rate (photons s^-1 pixel^-1)
    '''

    # convert apparent mag to surface flux 
    F=epsilon*zero_point*10**(-0.4*mu) #erg s^-1 cm^-2 A^-1 arcsec^-2

    # Find energy of effective photon
    h=6.626e-27 #erg s
    c=3e18 # Angstroms s^-1
    E_photon=h*c/WL_eff

    # find photon flux
    F_photon=F/E_photon #photons s^-1 cm^-2 Angstrom^-1 arcsec^-2

    # multiply by area of telescope and effective wavelength
    R_cm=100*D/2

    mu_b=F_photon*np.pi*R_cm**2*width_eff #photons s^-1 arcsec^-2

    # find platescale
    s=get_platescale(R,D) #arcsec micron^-1
    pixel_length_arcsec=pixel_length*s #arcsec 

    Rb=mu_b*pixel_length_arcsec**2 #photons s^-1 pixel^-1

    return Rb


if __name__=='__main__':

    # Problem 1

    Rs=0.2 #photons s^-1
    Rb=0.5 #photons s^-1 pixel^-1
    Rd=10/(60*60) #e s^-1 pixel^-1
    npix=4
    SNR=100
    Nr=5

    t=calculate_exposure_time(Rs,Rb,Rd,Nr,npix,SNR,guess=8000*60)[0]
    print("Problem 1 Number of Exposures: {:.1f}".format(t/60))


    # Problem 2
    D=2.3 #m
    R=2.1
    pixel_length=13.5 #microns
    m=22 #mag in V band
    mu=20 #background surface brightness (mag arcsec^-2)
    WL_eff=5450 #effective WL of V band (Angstroms)
    width_eff=880 #effective width of V band (Angstroms)
    zero_point=363.1*10**-11 #flux for 0 mag in V band (erg s^-1 cm^-2 A^-1)
    epsilon=0.9*0.7

    s=get_platescale(R,D)
    Rs=get_source_rate(m,zero_point,WL_eff,width_eff,D,epsilon)
    Rb=get_background_rate(mu,zero_point,WL_eff,width_eff,R,D,pixel_length,epsilon)

    print('Source Rate: {:.1f} photons s^-1'.format(Rs))
    print('Background Rate: {:.1f} photons s^-1 pixel^-1'.format(Rb))
    print('Platescale: {:.4f} arcsec micron^-1'.format(s))

    Rd=0
    Nr=4.5
    npix=4
    SNR=100
    t=calculate_exposure_time(Rs,Rb,Rd,Nr,npix,SNR,guess=1800)[0]
    print("Problem 2 Exposure Time: {:.1f} s".format(t))

    # When moon phase is new mu=22
    mu_newmoon=22
    Rb_newmoon=get_background_rate(mu_newmoon,zero_point,WL_eff,width_eff,R,D,pixel_length,epsilon)
    print('Background Rate for New Moon: {:.1f} photons s^-1 pixel^-1'.format(Rb_newmoon))

    t=calculate_exposure_time(Rs,Rb_newmoon,Rd,Nr,npix,SNR,guess=500)[0]
    print("Problem 2 New Moon Exposure Time: {:.1f} s".format(t))


