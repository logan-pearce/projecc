import numpy as np
import astropy.units as u
import astropy.constants as c

def update_progress(n,max_value):
    ''' Create a progress bar
    
    Args:
        n (int): current count
        max_value (int): ultimate values
    
    '''
    import sys
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    progress = np.round(float(n/max_value),decimals=2)
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    #if progress >= 1.:
    #    progress = 1
    #    status = "Done...\r\n"
    if n == max_value:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\r{0}% ({1} of {2}): |{3}|  {4}".format(np.round(progress*100,decimals=1), 
                                                  n, 
                                                  max_value, 
                                                  "#"*block + "-"*(barLength-block), 
                                                  status)
    sys.stdout.write(text)
    sys.stdout.flush()

def KeplersConstant(m1,m2):
    '''Compute Kepler's constant for two gravitationally bound masses k = G*m1*m2/(m1+m2) = G + (m1+m2)
        Inputs:
            m1,m2 (arr,flt): masses of the two objects in solar masses.  Must be astropy objects
        Returns:
            Kepler's constant in m^3 s^(-2)
    '''
    import astropy.constants as c
    import astropy.units as u
    m1 = m1.to(u.Msun)
    m2 = m2.to(u.Msun)
    mu = c.G*m1*m2
    m = (1/m1 + 1/m2)**(-1)
    kep = mu/m
    return kep.to((u.m**3)/(u.s**2))

def period(a,m):
    """ Given semi-major axis in AU and mass in solar masses, return period in years using Kepler's 3rd law"""
    import numpy as np
    import astropy.units as u
    try:
        a = a.to(u.AU)
        m = m.to(u.Msun)
    except:
        pass
    P = np.sqrt((a**3)/m)
    try:
        P = P.value
    except:
        pass
    return P

def PeriodToSMA(P,m):
    """ Given period in years and mass in solar masses, return semi-major axis in AU using Kepler's 3rd law"""
    import numpy as np
    import astropy.units as u
    try:
        P = P.to(u.yr)
        m = m.to(u.Msun)
    except:
        pass
    sma = ((P**2)*m) ** (1/3)
    try:
        sma = sma.value
    except:
        pass
    return sma


def MeanAnomToT0(MeanAnomaly, Period, RefEpoch = 2016, AfterDate = None):
    ''' Convert mean anomaly to epoch of periastron passage (T_0) for a given refence epoch.
    Args:
        MeanAnomaly (arr): Mean Anomaly in radians
        Period (arr): orbit period in years
        RefEpoch (flt): reference epoch for T0 in decimal years
        AfterDate (flt): T0 will be the first periastron passage after this date in decimal years

    Returns:
        arr: epoch of periastron passage in decimal years
    '''
    tau = MeanAnomaly / (2*np.pi)
    T0 = RefEpoch - (tau * Period)
    if AfterDate is not None:
        num_periods = (AfterDate - T0) / Period
        num_periods = np.ceil(num_periods)
        T0 += num_periods * Period
    return T0

def T0ToMeanAnom(T0, Period, RefEpoch = 2016):
    ''' Convert epoch of periastron passage (T0) to mean anomaly in radians
    Args:
        T0 (arr): epoch of periastron passage in decimal years
        Period (arr): orbit period in years
        RefEpoch (flt): reference epoch for T0 in decimal years
    Returns:
        arr: Mean Anomaly in radians
    '''
    tau = (RefEpoch - T0) / Period
    tau %= 1

    return tau * (2 * np.pi)

def NielsenPrior(Nsamples):
    ''' Prior on eccentricity derived from long period RV, from Nielsen et al. 2019
    '''
    e = np.linspace(1e-3,0.95,Nsamples)
    P = (2.1 - (2.2*e))
    P = P/np.sum(P)
    ecc = np.random.choice(e, size = Nsamples, p = P)
    return ecc

def DrawOrbits(number, EccNielsenPrior = False, DrawLON = True, DrawSMA = True, SMALowerBound = 0, SMAUpperBound = 3,
                FixedSMA = 100*u.AU):
    ''' Draw N sets of orbital elements from prior distributions:
            semi-major axis: LogUnif[LowerBound,UpperBound] (or fixed at 100 AU for OFTI application)
            eccentricity: Unif[0,1] or Linearly Descending
            inclination: cos(inc) Unif[-1,1]
            argument of periastron: Unif[0,2pi]
            longitude of nodes: Unif[0,2pi] (or fixed at 0 for OFTI application)
            mean anomaly: Unif[0,2pi]
    Args:
        number (int): Number of sets of elements to draw
        EccNielsenPrior (bool): if True, draw eccentricity from linearly descending prior. \
            If false, draw from uniform prior
        DrawLON (bool): If true, draw longitude of nodes from Unif[0,360] deg.  If false, \
            all LON values eill be zero
        DrawSMA (bool): If true, draw semi-major axis values from log uniform prior from \
            lower to upper bounds (in log space) in AU.  If false, all SMA values will be 100 AU.
        FixedSMA (astropy unit object): If DrawSMA = False, supply a value of SMA as an astropy unit object
    Returns:
        arr: array of N sma values in AU
        arr: array of eccentricity values
        arr: array of inclination values in degrees
        arr: array pf argument of periastron values in degrees
        arr: array of longitude of nodes values in degrees
        arr: array of mean anomaly values
    
    Written by Logan Pearce, 2019
    '''
    import astropy.units as u
    import numpy as np
    if DrawSMA:
        sma = 10 ** np.random.uniform(SMALowerBound,SMAUpperBound,number)
    else:
        sma = FixedSMA.to(u.AU)
        sma = np.array(np.linspace(sma,sma,number))
    # Eccentricity:
    if EccNielsenPrior:
        from projecc import NielsenPrior
        ecc = NielsenPrior(number)
    else:
        ecc = np.random.uniform(0.0,1.0,number)
    # Inclination in radians:
    cosi = np.random.uniform(-1.0,1.0,number)  #Draws sin(i) from a uniform distribution.  Inclination
    # is computed as the arccos of cos(i):
    inc = np.degrees(np.arccos(cosi))
    # Argument of periastron in degrees:
    argp = np.random.uniform(0.0,360.0,number)
    # Long of nodes:
    if DrawLON:
        lon = np.random.uniform(0.0,360.0,number)
    else:
        lon = np.degrees(0.0)
        lon = np.array([lon]*number)
    # orbit fraction (fraction of orbit completed at observation date since reference date)
    meananom = np.random.uniform(0.0,1.0,number) * 2 * np.pi
    return sma, ecc, inc, argp, lon, meananom

def EccentricityAnomaly(E,e,M):
    '''Eccentric anomaly function'''
    import numpy as np
    return E - (e*np.sin(E)) - M

def NRSolve(f, M0, e, h):
    ''' Newton-Raphson solver for eccentricity anomaly
    Inputs: 
        f (function): function to solve (transcendental ecc. anomaly function)
        M0 (float): mean anomaly
        e (float): eccentricity
        h (float): termination criteria for solver
    Returns: nextE (float): converged solution for eccentric anomaly
        Written by Logan Pearce, 2019
    '''
    import numpy as np
    from projecc import EccentricityAnomaly
    if M0 / (1.-e) - np.sqrt( ( (6.*(1-e)) / e ) ) <= 0:
        E0 = M0 / (1.-e)
    else:
        E0 = (6. * M0 / e) ** (1./3.)
    lastE = E0
    nextE = lastE + 10* h 
    number=0
    while (abs(lastE - nextE) > h) and number < 1001: 
        new = f(nextE,e,M0) 
        lastE = nextE
        nextE = lastE - new / (1.-e*np.cos(lastE)) 
        number=number+1
        if number >= 1000:
            nextE = float('NaN')
    return nextE

def DanbySolve(f, M0, e, h, maxnum=50):
    ''' Newton-Raphson solver for eccentricity anomaly based on "Danby" method in 
        Wisdom textbook
    Inputs: 
        f (function): function to solve (transcendental ecc. anomaly function)
        M0 (float): mean anomaly
        e (float): eccentricity
        h (float): termination criteria for solver
        maxnum (int): if it takes more than maxnum iterations,
            use the Mikkola solver instead.
    Returns: nextE (float): converged solution for eccentric anomaly
        Written by Logan Pearce, 2020
    '''
    import numpy as np
    from projecc import EccentricityAnomaly
    #f = EccentricityAnomaly
    k = 0.85
    E0 = M0 + np.sign(np.sin(M0))*k*e
    lastE = E0
    nextE = lastE + 10* h 
    number=0
    delta_D = 1
    while (delta_D > h) and number < maxnum+1: 
        fx = f(nextE,e,M0) 
        fp = (1.-e*np.cos(lastE)) 
        fpp = e*np.sin(lastE)
        fppp = e*np.cos(lastE)
        lastE = nextE
        delta_N = -fx / fp
        delta_H = -fx / (fp + 0.5*fpp*delta_N)
        delta_D = -fx / (fp + 0.5*fpp*delta_H + (1./6)*fppp*delta_H**2)
        nextE = lastE + delta_D
        number=number+1
        if number >= maxnum:
            from projecc import mikkola_solve
            nextE = mikkola_solve(M0,e)
    return nextE

def MikkolaSolve(M,e):
    ''' Analytic solver for eccentricity anomaly from Mikkola 1987. Most efficient
        when M near 0/2pi and e >= 0.95.
    Inputs: 
        M (float): mean anomaly
        e (float): eccentricity
    Returns: eccentric anomaly
        Written by Logan Pearce, 2020
    '''
    # Constants:
    alpha = (1 - e) / ((4.*e) + 0.5)
    beta = (0.5*M) / ((4.*e) + 0.5)
    ab = np.sqrt(beta**2. + alpha**3.)
    z = np.abs(beta + ab)**(1./3.)

    # Compute s:
    s1 = z - alpha/z
    # Compute correction on s:
    ds = -0.078 * (s1**5) / (1 + e)
    s = s1 + ds

    # Compute E:
    E0 = M + e * ( 3.*s - 4.*(s**3.) )

    # Compute final correction to E:
    sinE = np.sin(E0)
    cosE = np.cos(E0)

    f = E0 - e*sinE - M
    fp = 1. - e*cosE
    fpp = e*sinE
    fppp = e*cosE
    fpppp = -fpp

    dx1 = -f / fp
    dx2 = -f / (fp + 0.5*fpp*dx1)
    dx3 = -f / ( fp + 0.5*fpp*dx2 + (1./6.)*fppp*(dx2**2) )
    dx4 = -f / ( fp + 0.5*fpp*dx3 + (1./6.)*fppp*(dx3**2) + (1./24.)*(fpppp)*(dx3**3) )

    return E0 + dx4

def RotateZ(vector,theta):
    """ Rotate a 3D vector about the +z axis
        Inputs:
            vector: 3d vector array
            theta [rad]: angle to rotate the vector about
        Returns: rotated vector
    """
    import numpy as np
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
               [np.sin(theta), np.cos(theta), 0],
               [0, 0, 1]
              ], dtype = object)
    if np.ndim(vector) == 1:
        out = np.zeros(3)
        out[0] = R[0,0]*vector[0] + R[0,1]*vector[1] + R[0,2]*vector[2]
        out[1] = R[1,0]*vector[0] + R[1,1]*vector[1] + R[1,2]*vector[2]
        out[2] = R[2,0]*vector[0] + R[2,1]*vector[1] + R[2,2]*vector[2]
        
    else:
        out = np.zeros((3,vector.shape[1]))
        out[0] = R[0,0]*vector[0,:] + R[0,1]*vector[1,:] + R[0,2]*vector[2,:]
        out[1] = R[1,0]*vector[0,:] + R[1,1]*vector[1,:] + R[1,2]*vector[2,:]
        out[2] = R[2,0]*vector[0,:] + R[2,1]*vector[1,:] + R[2,2]*vector[2,:]
    
    return out

def RotateX(vector,theta):
    """ Rotate a 3D vector about the +x axis
        Inputs:
            vector: 3d vector array
            theta [rad]: angle to rotate the vector about
        Returns: rotated vector
    """
    import numpy as np
    if np.ndim(vector) == 1:
        R = np.array([[1., 0., 0.],
              [0., np.cos(theta), -np.sin(theta)],
              [0., np.sin(theta), np.cos(theta)]  
              ], dtype = object)
        out = np.zeros(3)
        out[0] = R[0,0]*vector[0] + R[0,1]*vector[1] + R[0,2]*vector[2]
        out[1] = R[1,0]*vector[0] + R[1,1]*vector[1] + R[1,2]*vector[2]
        out[2] = R[2,0]*vector[0] + R[2,1]*vector[1] + R[2,2]*vector[2]
        
    else:
        R = np.array([[[1.]*len(theta), 0., 0.],
              [0., np.cos(theta), -np.sin(theta)],
              [0., np.sin(theta), np.cos(theta)]  
              ], dtype = object)
        out = np.zeros((3,vector.shape[1]))
        out[0] = R[0,0]*vector[0,:] + R[0,1]*vector[1,:] + R[0,2]*vector[2,:]
        out[1] = R[1,0]*vector[0,:] + R[1,1]*vector[1,:] + R[1,2]*vector[2,:]
        out[2] = R[2,0]*vector[0,:] + R[2,1]*vector[1,:] + R[2,2]*vector[2,:]
    return out

def KeplerianToCartesian(sma,ecc,inc,argp,lon,meananom,kep, solvefunc = DanbySolve):
    """ Given a set of Keplerian orbital elements, returns the observable 3-dimensional position, velocity, 
        and acceleration at the specified time.  Accepts and arbitrary number of input orbits.  Semi-major 
        axis must be an astropy unit object in physical distance (ex: au, but not arcsec).  The observation
        time must be converted into mean anomaly before passing into function.
        Inputs:
            sma (1xN arr flt) [au]: semi-major axis in au, must be an astropy units object
            ecc (1xN arr flt) [unitless]: eccentricity
            inc (1xN arr flt) [deg]: inclination
            argp (1xN arr flt) [deg]: argument of periastron
            lon (1xN arr flt) [deg]: longitude of ascending node
            meananom (1xN arr flt) [radians]: mean anomaly 
            kep (1xN arr flt): kepler constant = mu/m where mu = G*m1*m2 and m = [1/m1 + 1/m2]^-1 . 
                        In the limit of m1>>m2, mu = G*m1 and m = m2
        Returns:
            pos (3xN arr) [au]: position in xyz coords in au, with 
                        x = pos[0], y = pos[1], z = pos[2] for each of N orbits
                        +x = +Dec, +y = +RA, +z = towards observer
            vel (3xN arr) [km/s]: velocity in xyz plane.
            acc (3xN arr) [km/s/yr]: acceleration in xyz plane.
        Written by Logan Pearce, 2019, inspired by Sarah Blunt
    """
    import numpy as np
    import astropy.units as u
    
    try:
        sma = sma.to(u.AU)
    except:
        sma = sma*u.AU
    
    # Compute mean motion and eccentric anomaly:
    meanmotion = np.sqrt(kep / sma**3).to(1/u.s)
    try:
        E = solvefunc(EccentricityAnomaly, meananom, ecc, 0.001)
    except:
        nextE = [solvefunc(EccentricityAnomaly, varM,vare, 0.001) for varM,vare in zip(meananom, ecc)]
        E = np.array(nextE)

    # Compute position:
    try:
        pos = np.zeros((3,len(sma)))
    # In the plane of the orbit:
        pos[0,:], pos[1,:] = (sma*(np.cos(E) - ecc)).value, (sma*np.sqrt(1-ecc**2)*np.sin(E)).value
    except:
        pos = np.zeros(3)
        pos[0], pos[1] = (sma*(np.cos(E) - ecc)).value, (sma*np.sqrt(1-ecc**2)*np.sin(E)).value
        
    # Rotate to plane of the sky:
    pos = RotateZ(pos, np.radians(argp))
    pos = RotateX(pos, np.radians(inc))
    pos = RotateZ(pos, np.radians(lon))
    
    # compute velocity:
    try:
        vel = np.zeros((3,len(sma)))
        vel[0], vel[1] = (( -meanmotion * sma * np.sin(E) ) / ( 1 - ecc * np.cos(E) )).to(u.km/u.s).value, \
                (( meanmotion * sma * np.sqrt(1 - ecc**2) *np.cos(E) ) / ( 1 - ecc * np.cos(E) )).to(u.km/u.s).value
    except:
        vel = np.zeros(3)
        vel[0], vel[1] = (( -meanmotion * sma * np.sin(E) ) / ( 1- ecc * np.cos(E) )).to(u.km/u.s).value, \
                (( meanmotion * sma * np.sqrt(1 - ecc**2) *np.cos(E) ) / ( 1 - ecc * np.cos(E) )).to(u.km/u.s).value
    vel = RotateZ(vel, np.radians(argp))
    vel = RotateX(vel, np.radians(inc))
    vel = RotateZ(vel, np.radians(lon))
    
    # Compute accelerations numerically
    # Generate a nearby future time point(s) along the orbit:
    deltat = 0.002*u.yr
    try:
        acc = np.zeros((3,len(sma)))
        futurevel = np.zeros((3,len(sma)))
    except:
        acc = np.zeros(3)
        futurevel = np.zeros(3)
    # Compute new mean anomaly at future time:
    futuremeananom = meananom + meanmotion*((deltat).to(u.s))
    # Compute new eccentricity anomaly at future time:
    try:
        futureE = [solvefunc(EccentricityAnomaly, varM,vare, 0.001) for varM,vare in zip(futuremeananom.value, ecc)]
        futureE = np.array(futureE)
    except:
        futureE = solvefunc(EccentricityAnomaly, futuremeananom.value, ecc, 0.001)
    # Compute new velocity at future time:
    futurevel[0], futurevel[1] = (( -meanmotion * sma * np.sin(futureE) ) / ( 1- ecc * np.cos(futureE) )).to(u.km/u.s).value, \
                (( meanmotion * sma * np.sqrt(1 - ecc**2) *np.cos(futureE) ) / ( 1 - ecc * np.cos(futureE) )).to(u.km/u.s).value
    futurevel = RotateZ(futurevel, np.radians(argp))
    futurevel = RotateX(futurevel, np.radians(inc))
    futurevel = RotateZ(futurevel, np.radians(lon))
    acc = (futurevel-vel)/deltat.value
    
    return np.transpose(pos)*u.au, np.transpose(vel)*(u.km/u.s), np.transpose(acc)*(u.km/u.s/u.yr)

def CartesianToKeplerian(pos, vel, kep):
    """Given observables XYZ position and velocity, compute orbital elements.  Position must be in
       au and velocity in km/s.  Returns astropy unit objects for all orbital elements.
        Args:
            pos (3xN arr) [au]: position in xyz coords in au, with 
                        x = pos[0], y = pos[1], z = pos[2] for each of N orbits
                        +x = +Dec, +y = +RA, +z = towards observer
                        Must be astropy unit array e.g: [1,2,3]*u.AU, ~NOT~ [1*u.AU,2*u.AU,3*u,AU]
            vel (3xN arr) [km/s]: velocity in xyz plane.  Also astropy unit array
            kep (flt) [m^3/s^2] : kepler's constant.  From output of orbittools.keplersconstant. Must be
                        astropy unit object.
        Returns:
            sma (1xN arr flt) [au]: semi-major axis in au, must be an astropy units object
            ecc (1xN arr flt) [unitless]: eccentricity
            inc (1xN arr flt) [deg]: inclination
            argp (1xN arr flt) [deg]: argument of periastron
            lon (1xN arr flt) [deg]: longitude of ascending node
            meananom (1xN arr flt) [radians]: mean anomaly 
        Written by Logan Pearce, 2019, inspired by Sarah Blunt
    """
    import numpy as np
    import astropy.units as u
    # rvector x vvector:
    rcrossv = np.cross(pos, vel)
    # specific angular momentum:
    h = np.sqrt(rcrossv[0]**2 + rcrossv[1]**2 + rcrossv[2]**2)
    # normal vector:
    n = rcrossv / h
    
    # inclination:
    inc = np.arccos(n[2])
    
    # longitude of ascending node:
    lon = np.arctan2(n[0],-n[1])
    
    # semi-major axis:
    r = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
    v = np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
    sma = 1/((2./r) - ((v)**2/kep))
    
    # ecc and f:
    rdotv = pos[0]*vel[0] + pos[1]*vel[1] + pos[2]*vel[2]
    rdot = rdotv/r
    parameter = h**2 / kep
    ecosf = parameter/r - 1
    esinf = (h)*rdot / (kep.to(u.m**3/u.s**2))
    ecc = np.sqrt(ecosf**2 + esinf**2)
    f = np.arctan2(esinf,ecosf)
    f = f.value%(2.*np.pi)
    
    # E and M:
    E = 2. * np.arctan( np.sqrt( (1 - ecc.value)/ (1 + ecc.value) ) * ( np.tan(f/2.) ) )
    M = E - ecc * np.sin(E)
    
    # argument of periastron:
    rcosu = pos[0] * np.cos(lon) + pos[1] * np.sin(lon)
    rsinu = (-pos[0] * np.sin(lon) + pos[1] * np.cos(lon)) / np.cos(inc)
    uangle = np.arctan2(rsinu,rcosu)
    argp = uangle.value - f
    
    return sma.to(u.au), ecc, np.degrees(inc), (np.degrees(argp)%360.)*u.deg, (np.degrees(lon.value)%360.)*u.deg, M


def GetOrbitTracks(sma,ecc,inc,argp,lon,kep, solvefunc = DanbySolve, Npoints = 100):
    ''' For a set of orbital parameters, compute sky plane positions of points along the entire orbit.
    '''
    Ms = np.linspace(0,2*np.pi,Npoints)
    Xs = np.zeros(Npoints)
    Ys = np.zeros(Npoints)
    for i in range(Npoints):
        pos, vel, acc = KeplerianToCartesian(sma,ecc,inc,argp,lon,Ms[i],kep, solvefunc = solvefunc)
        Xs[i] = pos[0].value
        Ys[i] = pos[1].value
    return Xs, Ys

def DrawSepAndPA(Nsamples, Mstar1, Mstar2, SMALogLowerBound = 0, SMALogUpperBound = 3,
                EccNielsenPrior = True, DrawLON = True, 
                DrawSMA = True, solvefunc = DanbySolve, FixedSMA =100*u.AU):
    ''' Generate a set of Nsamples simulated companions and return their current separation \
        and position angle in the plane of the sky.

    Args:
        Nsamples (int): number of simulated companions to generate
        Mstar1 (astropy units object): Mass of primary
        Mstar2 (astropy units object): Mass of simulated companion
        EccNielsenPrior (bool): If true, draw eccentricity from a linearly descending prior given in Nielsen+ 2019,
            if false draw from uniform prior on [0,1].  Default = True
        DrawLON (bool): If true, draw longitude of nodes from Unif[0,360] deg.  If false all LON values will be zero.
            Default = True
        DrawSMA (bool): If true, draw semi-major axis from log linear prior.  if false all SMA will be 100 AU. Default = True
        SMALogLowerBound, SMALogUpperBound (flt): draw semi-major axis from log linear prior between these
            two bounds.  Default = 0,3
        solvefunc (function): Function to use for solving for eccentricity anomaly.  Default = DanbySolve
        FixedSMA (astropy unit object): If DrawSMA = False, supply a value of SMA as an astropy unit object

    Returns:
        arr: separations in AU
        arr: position angles in degrees east of north

    '''
    from projecc import DrawOrbits, DanbySolve, KeplerianToCartesian
    import numpy as np

    kep = KeplersConstant(Mstar1,Mstar2)

    sma, ecc, inc, argp, lon, meananom = DrawOrbits(Nsamples, EccNielsenPrior = EccNielsenPrior, DrawLON = DrawLON, 
            DrawSMA = DrawSMA, SMALowerBound = SMALogLowerBound, SMAUpperBound = SMALogUpperBound, FixedSMA = FixedSMA)

    pos, vel, acc = KeplerianToCartesian(sma,ecc,inc,argp,lon,meananom,kep, solvefunc = solvefunc)
    r = np.sqrt(pos[:,0]**2 + pos[:,1]**2).value
    phi = ((np.degrees(np.arctan2(pos[:,1].value,pos[:,0].value))) ) % 360

    return r, phi

def GetSepAndPA(pos):
    ''' For the output from KeplerianToCartesian, compute the separation and position angle for the results
    '''
    r = np.sqrt(pos[:,0]**2 + pos[:,1]**2).value
    dec = pos[:,0]
    ra = pos[:,1]
    phi = (np.arctan2(ra,dec).to(u.deg).value)%360
    return r, phi

def MonteCarloIt(thing, N = 10000):
    ''' 
    Generate a random sample of size = N from a 
    Gaussian centered at thing[0] with std thing[1]
    
    Args:
        thing (tuple, flt): tuple of (value,uncertainty).  Can be either astropy units object \
            or float
        N (int): number of samples
    Returns:
        array: N random samples from a Gaussian.

    Written by Logan Pearce, 2020
    '''
    try:
        out = np.random.normal(thing[0].value,thing[1].value,N)
    except:
        out = np.random.normal(thing[0],thing[1],N)

    return out


def GetPhaseAngle(MeanAnom, ecc, inc, argp):
    ''' Function for returning observed phase angle given orbital elements
    Args:
        MeanAnom (flt): Mean anomly in radians, where MeanAnom = orbit fraction*2pi, or M=2pi * time/Period
        ecc (flt): eccentricity, defined on [0,1)
        inc (flt): inclination in degrees, where inc = 90 is edge on, inc = 0 or 180 is face on orbit
        argp (flt): argument of periastron in degrees, defined on [0,360)
        
    Returns:
        flt: phase angle in degrees
    Written by Logan Pearce, 2023
    '''
    import numpy as np
    inc = np.radians(inc)
    argp = np.radians(argp)
    EccAnom = DanbySolve(EccentricityAnomaly, MeanAnom, ecc, 0.001, maxnum=50)
    TrueAnom = 2*np.arctan( np.sqrt( (1+ecc)/(1-ecc) ) * np.tan(EccAnom/2) )
    Alpha = np.arccos( np.sin(inc) * np.sin(TrueAnom + argp) )
    
    return np.degrees(Alpha)

def GetPhasesFromOrbit(sma,ecc,inc,argp,lon,Ms,Mp):
    ''' Creates an array of viewing phases for an orbit in the plane of the sky to the observer with the maximum phase
     (faintest contrast) at inferior conjunction (where planet is aligned between star and observer) and minimum phase 
     (brightest) at superior conjunction.

    args:
        sma [flt]: semi-major axis in au 
        ecc [flt]: eccentricity
        inc [flt]: inclination in degrees
        argp [flt]: argument of periastron in degrees
        lon [flt]: longitude of ascending node in degrees
        Ms [flt]: star mass in solar masses
        Mp [flt]: planet mass in Jupiter masses

    returns:
        arr: array of viewing phases from periastron back to periastron.

    '''
    # Getting phases for the orbit described by the mean orbital params:
    import astropy.units as u
    from myastrotools.tools import keplerian_to_cartesian, keplersconstant
    # Find the above functions here: https://github.com/logan-pearce/myastrotools/blob/2bbc284ab723d02b7a7189494fd3eabaed434ce1/myastrotools/tools.py#L2593
    # and here: https://github.com/logan-pearce/myastrotools/blob/2bbc284ab723d02b7a7189494fd3eabaed434ce1/myastrotools/tools.py#L239
    # Make lists to hold results:
    xskyplane,yskyplane,zskyplane = [],[],[]
    phase = []
    # How many points to compute:
    Npoints = 1000
    # Make an array of mean anomaly:
    meananom = np.linspace(0,2*np.pi,Npoints)
    # Compute kepler's constant:
    kepmain = keplersconstant(Ms*u.Msun, Mp*u.Mjup)
    # For each orbit point:
    for m in meananom:
        # compute 3d projected position:
        pos, vel, acc = keplerian_to_cartesian(sma*u.au,ecc,inc,argp,lon,m,kepmain)
        # add to list:
        xskyplane.append(pos[0].value)
        yskyplane.append(pos[1].value)
        zskyplane.append(pos[2].value)

    ##### Getting the phases as a function of mean anom: ###########
    ###### Loc of inf conj:
    # Find all points with positive z -> out of sky plane:
    towardsobsvers = np.where(np.array(zskyplane) > 0)[0]
    # mask everything else:
    maskarray = np.ones(Npoints) * 99999
    maskarray[towardsobsvers] = 1
    # mask x position:
    xtowardsobsvers = np.array(xskyplane)*maskarray
    # find where x position is minimized in the section of orbit towards the observer:
    infconj_ind = np.where( np.abs(xtowardsobsvers) == min(np.abs(xtowardsobsvers)) )[0][0]
    ###### Loc of sup conj:
    # Do the opposite - find where x in minimized for points into the plane/away from observer
    awayobsvers = np.where(np.array(zskyplane) < 0)[0]
    maskarray = np.ones(Npoints) * 99999
    maskarray[awayobsvers] = 1
    xawayobsvers = np.array(xskyplane)*maskarray
    supconj_ind = np.where( np.abs(xawayobsvers) == min(np.abs(xawayobsvers)) )[0][0]

    #### Find max and min value phases for this inclination:
    phis = np.linspace(0,180,Npoints)
    phases = np.array(alphas(inc,phis))
    minphase = min(phases)
    maxphase = max(phases)
    # Generate empty phases array:
    phases_array = np.ones(Npoints)

    ###### Set each side of the phases array to range from min to max phase on either side of 
    # inf/sup conjunctions:
    if supconj_ind > infconj_ind:
        # Set one side of the phases array to phases from max to min
        phases_array[0:len(xskyplane[infconj_ind:supconj_ind])] = np.linspace(maxphase,minphase,
                                                            len(xskyplane[infconj_ind:supconj_ind]))
        # # Set the other side to phases from min to max
        phases_array[len(xskyplane[infconj_ind:supconj_ind]):] = np.linspace(minphase,maxphase,
                                                            len(xskyplane)-len(xskyplane[infconj_ind:supconj_ind]))
        # Finally roll the array to align with the mean anomaly array:
        phases_array = np.roll(phases_array,infconj_ind)


    else:
        # Set one side of the phases array to phases from min to max
        phases_array[0:len(xskyplane[supconj_ind:infconj_ind])] = np.linspace(minphase,maxphase,
                                                            len(xskyplane[supconj_ind:infconj_ind]))
        # # Set the other side to phases from max to min
        phases_array[len(xskyplane[supconj_ind:infconj_ind]):] = np.linspace(maxphase,minphase,
                                                            len(xskyplane)-len(xskyplane[supconj_ind:infconj_ind]))
        # Finally roll the array to align with the mean anomaly array:
        phases_array = np.roll(phases_array,supconj_ind)

    return xskyplane, yskyplane, zskyplane, phases_array


def GetKDE(ra, dec, size=100j):
    from scipy import stats
    xmin = ra.min()
    xmax = ra.max()
    ymin = dec.min()
    ymax = dec.max()
    X, Y = np.mgrid[xmin:xmax:size, ymin:ymax:size]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([ra,dec])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    return Z.T, xmin, ymin, xmax, ymax

def GetCLevels(arr, sigmas = [1,2,3]):
    Hflat = arr.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]

    levels = 1.0 - np.exp(-0.5 * np.array(sigmas) ** 2)
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except IndexError:
            V[i] = Hflat[0]
    clevels = np.sort(V)
    return clevels

from scipy.ndimage import gaussian_filter
def GetHist(xx,yy, sigmas = [1,2]):
    ''' For arrays xx and yy, generate a 2d histogram and values for plotting contours
    '''
    H, xe,ye = np.histogram2d(xx, yy, bins=(100,100))
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    
    levels = 1.0 - np.exp(-0.5 * np.array(sigmas) ** 2)
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except IndexError:
            V[i] = Hflat[0]
    midpoints = (xe[1:] + xe[:-1])/2, (ye[1:] + ye[:-1])/2
    return H, xe, ye, midpoints, np.sort(V)

def alphas(inc, phis):
    '''
    From Lovis+ 2017 sec 2.1:<br>
    $\cos(\alpha) = - \sin(i) \cos(\phi)$<br>
    where $i$ is inclination and $\phi$ is orbital phase with $\phi= 0$ at inferior conjunction

    args:
        inc [flt]: inclination in degrees
        phis [arr]: array of phi values from zero to 360 in degrees

    returns:
        arr: array of viewing phase angles for an orbit from inferior conjunction back to 
            inferior conjunction
    '''
    alphs = []
    for phi in phis:
        alphs.append(np.degrees(np.arccos(-np.sin(np.radians(inc)) * np.cos(np.radians(phi)))))
    return alphs


def GetOrbitPlaneOfSky(sma,ecc,inc,argp,lon,meananom,kep):
    ''' For a value fo sma, ecc, inc, argp, lan, and mass, compute the position in the sky plane for one or
    an array of mean anomaly values past periastron.

    args:
        sma [astropy unit object]: semi-major axis in au
        ecc [flt]: eccentricity
        inc [flt]: inclination in degrees
        argp [flt]: argument of periastron in degrees
        lon [flt]: longitude of nodes in degrees
        meananom [flt or arr]: a single value or array of values for the mean anomaly in radians at which 
            to compute positions
        kep [astropy unit object]: value of Kepler's constant for the system
    
    returns:
        flt or arr: X value of position, where +x corresponds to +Declination
        flt or arr: Y value, where +Y corresponds to +Right Ascension
        flt or arr: Z value, where +Z corresponds to towards the observer
    '''
    try:
        X,Y,Z = [],[],[]
        for m in meananom:
            pos, vel, acc = KeplerianToCartesian(sma,ecc,inc,argp,lon,m,kep)
            X.append(pos[0].value)
            Y.append(pos[1].value)
            Z.append(pos[2].value)
    except TypeError:
        X = pos[0].value
        Y = pos[1].value
        Z = pos[2].value
    return X, Y, Z

def GetOrbitPlaneOfOrbit(sma,ecc,meananom,kep):
    ''' For a value fo sma, ecc, and mass, compute the position in the orbit plane for one or
    an array of mean anomaly values past periastron.

    args:
        sma [astropy unit object]: semi-major axis in au
        ecc [flt]: eccentricity
        meananom [flt or arr]: a single value or array of values for the mean anomaly in radians at which 
            to compute positions
        kep [astropy unit object]: value of Kepler's constant for the system
    
    returns:
        flt or arr: x value of position, where +x corresponds to semi-major axis towards periastron
        flt or arr: y value, where +y corresponds to semi-minor axis counterclockwise perpendiculat to +x
        flt or arr: z value, where +z corresponds to angular momentum vector for right handed system
    '''
    import numpy as np
    import astropy.units as u
    try:
        x,y,z = [],[],[]
        for m in meananom:
            E = DanbySolve(EccentricityAnomaly, m, ecc, 0.001)
            x.append((sma*(np.cos(E) - ecc)).value)
            y.append((sma*np.sqrt(1-ecc**2)*np.sin(E)).value)
            z.append(0)
    except TypeError:
        E = DanbySolve(EccentricityAnomaly, meananom, ecc, 0.001)
        x = (sma*(np.cos(E) - ecc)).value
        y = (sma*np.sqrt(1-ecc**2)*np.sin(E)).value
        z = 0
        
    return x,y,z

def ComputeTeq(StarTeff, StarRad, sep, Ab = 0.3, fprime = 1/4):
    ''' Compute average equilibrium temperature from flux balance, from Seager 2016 Exoplanet Atmospheres eqn 3.9
    https://books.google.com/books?id=XpaYJD7IE20C

    Args:
        StarTeff [flt]: star effective temperature
        StarRad [astropy unit object]: star radius
        sep [astropy unit object]: star-planet orbit plane separation in distance units
        Ab [flt]: Albedo. Default = 0.3
        fprime [flt]: Default = 1/4
    Returns
        flt: average planet equilibrium temperature
    '''
    import astropy.units as u
    StarRad = StarRad.to(u.km)
    sep = sep.to(u.km)
    return (StarTeff * np.sqrt(StarRad/sep) * ((fprime * (1 - Ab))**(1/4))).value

from astropy.time import Time
class Planet(object):
    ''' Class for simulating orbits in the plane of the sky at a specific time given orbital parameters with gaussian errors.
    All parameters should be supplied as a tuple of [mean,std]
    
    Args:
        sma [tuple]: semi-major axis in au
        ecc [tuple]: eccentricity
        argp [tuple]: argument of periastron of planet in degrees
        lan [tuple]: longitude of ascending node in degrees
        period [tuple]: period in days
        Mpsini [tuple]: planet minimum mass in Earth masses
        Mstar [tuple]: star mass in solar masses
        distance [tuple]: system distance in parsec
        obsdate [astropy Time object]: date around which you'd like to observe
        Mp_is_Mpsini [bool]: Set to True if the planet mass tuple is an Msin(i) value, Default: True
    '''
    def __init__(self,sma,ecc,inc,argp,lan,period,t0,Mpsini,Mstar,parallax,dec, Mp_is_Mpsini = True):
        self.sma = sma
        self.ecc = ecc
        self.inc = inc
        self.argp = argp
        self.lan = lan
        self.period = period
        t01 = Time(t0[0], format='jd')
        self.t0 = [t01.decimalyear, t0[1]*u.d.to(u.yr)]
        self.Mp_is_Mpsini = Mp_is_Mpsini
        if Mp_is_Mpsini:
            self.Mpsini = Mpsini
        else:
            self.Mp = Mpsini
        self.dec = dec
        self.Mstar = Mstar
        self.parallax = parallax
        distance = 1000/(MonteCarloIt(parallax))
        self.distance = [np.mean(distance),np.std(distance)]
        self.GetDateOfMaxElongation()
        
    def GetDateOfMaxElongation(self):
        # Find some periastron dates going forward from t0
        timespan = np.arange(0,1000,1) # days
        tps = [self.t0[0]]
        for t in timespan:
            pps = MonteCarloIt(self.period)
            tps.append(tps[-1] + pps*u.d.to(u.yr))
        tps_mean = np.array([np.mean(t) for t in tps])
        tps_std = np.array([np.std(t) for t in tps])
        self.periastron_times = tps_mean
        self.periastron_times_error = tps_std
        # For a time in the near future:
        obstime = Time('2025-04-15T00:00:00', format='isot').decimalyear
        self.obstime = obstime
        # find the most recent time of periastron to that date:
        ind = np.where(self.periastron_times <= obstime)[0]
        nearest_periastron = tps_mean[ind[-1]]
        self.nearest_periastron_to_Apr152025 = nearest_periastron
        # Using the mean orbit parameter values generate an array of separations spanning one orbit:
        Orbfrac = np.linspace(0,1,1000)
        # create empty container:
        seps1 = []
        if type(self.inc) == list:
            incl = self.inc[0]
        elif type(self.inc) == int:
            incl = self.inc
        else:
            incl = 60
        if type(self.lan) == list:
            lan = self.lan[0]
        elif type(self.lan) == int:
            lan = self.lan
        else:
            lan = 0
        # Compute kepler's constant depending on if provided mass is Mpsini or Mp:
        try:
            kep = KeplersConstant(self.Mstar[0]*u.Msun,(self.Mpsini[0]/np.sin(np.radians(incl)))*u.Mearth)
        except AttributeError:
            kep = KeplersConstant(self.Mstar[0]*u.Msun,self.Mp[0]*u.Mearth)
        ras1, decs1 = [], []
        for i in range(len(Orbfrac)):
            pos, vel, acc = KeplerianToCartesian(self.sma[0],
                                                 self.ecc[0],
                                                 incl,
                                                 self.argp[0],
                                                 lan,
                                                 Orbfrac[i]*2*np.pi,kep, solvefunc = DanbySolve)
            decs1.append((pos[0].value / self.distance[0])*1000)
            ras1.append((pos[1].value / self.distance[0])*1000)
            seps1.append((np.sqrt(pos[0].value**2 + pos[1].value**2)/ self.distance[0])*1000)
        self.ras_mean_params = ras1
        self.decs_mean_params = decs1
        self.seps_mean_params = seps1
        
        # Find where its at largest separation:
        self.ind_of_max_elongation = np.where(seps1 == max(seps1))[0]
        # What fraction on the period is that?
        self.time_of_max_elongation_days = Orbfrac[self.ind_of_max_elongation] * self.period[0]
        # add that time to the nearest periastron and that gives the time of max elongation:
        self.date_of_max_elongation = self.nearest_periastron_to_Apr152025 + self.time_of_max_elongation_days*u.d.to(u.yr)
        

class OrbitSim(object):
    def __init__(self, planet, date, Ntrials = 100000, limit_inc_lt90 = True):
        ''' Generate an array of points along an orbit at a specific date for a Planet object
        
        Args:
            planet [Planet object]:
            date [decimal year]: date at which to generate points
            Ntrials [int]: number of points to generate
            limit_inc_lt90 [bool]: if True, inc will be generated from cos(i) unif in [0,1], else inc will be 
                             generated from cos(i) unif in [-1,1]
        
        Returns:
            object with attributes of arrays of size Ntrials drawn from normal distrbutions around provided 
            tuples for each of the following parameters:
                sma in au
                ecc
                inc
                argp
                lan
                meananomaly at date
                kepler's constant (mass parameter)
                Mstar
                Mplanet
                distance
                period
        '''
        self.sma = MonteCarloIt(planet.sma, N = Ntrials)
        self.ecc = MonteCarloIt(planet.ecc, N = Ntrials)
        self.argp = MonteCarloIt(planet.argp, N = Ntrials)
        self.period = MonteCarloIt([planet.period[0]*u.d.to(u.yr),
                                    planet.period[1]*u.d.to(u.yr)],
                                    N = Ntrials)
        self.Mstar = MonteCarloIt(planet.Mstar, N = Ntrials)
        self.distance = MonteCarloIt(planet.distance, N = Ntrials)
        
        try:
            # if inc = nan
            if np.isnan(planet.inc):
                # draw inc from cos(i) uniform dist:
                if limit_inc_lt90:
                    cosi = np.random.uniform(0.09,0.985, Ntrials)
                else:
                    cosi = np.random.uniform(-0.985,0.985,Ntrials)
                self.inc = np.degrees(np.arccos(cosi))
            # if inc is single value:
            elif type(planet.inc) == int or type(planet.inc) == float:
                # just make an array at that value
                self.inc = np.array([planet.inc]*Ntrials)
            # if inc is a tuple:
            elif type(planet.inc) == list:
                # draw from a normal distribution
                self.inc = MonteCarloIt(planet.inc, N = Ntrials)
        except ValueError:
            self.inc = MonteCarloIt(planet.inc, N = Ntrials)
        
        # if the planet mass entered is an Msini:
        if planet.Mp_is_Mpsini:
            # draw from normal distribution:
            self.Mpsini = MonteCarloIt(planet.Mpsini, N = Ntrials)
            # and divide by sin(i)
            self.Mp = self.Mpsini / np.sin(np.radians(self.inc))
        else:
            # otherwise draw from normal distribiton:
            self.Mp = MonteCarloIt(planet.Mp, N = Ntrials)

        try:
            if np.isnan(planet.lan):
                self.lan = np.random.uniform(size = Ntrials) * 360
            elif type(planet.lan) == int or type(planet.lan) == float:
                self.lan = np.array([planet.lan]*Ntrials)
        except ValueError:
            self.lan = MonteCarloIt(planet.lan, N = Ntrials)
        # find the most recent periastron time:
        ind = np.where(planet.periastron_times <= date)[0]
        # Add error on t0 to error on periastron passage date in quadrature:
        t0_error = np.sqrt(planet.t0[1]**2 + planet.periastron_times_error[ind[-1]]**2)
        # make MonetCarlo array
        trefarray = MonteCarloIt([planet.periastron_times[ind[-1]],
                                t0_error],
                                N = Ntrials) # in decimalyear
        self.trefarray = trefarray
        # compute amount of time from periastron to obsdate:
        deltaT = (date - trefarray) # in decimalyear
        self.deltaT = deltaT
        # How many periods is that?
        Nperiods = deltaT / (planet.period[0]*u.d.to(u.yr))
        self.Nperiods = Nperiods
        #Mean anomaly is Nperiods time 2pi:
        self.Meananom = (Nperiods % 1) * 2*np.pi
        # Compute kepler's constant:
        self.kep = KeplersConstant(self.Mstar*u.Msun,self.Mp*u.Mearth)
        # compute position:
        pos, vel, acc = KeplerianToCartesian(self.sma,self.ecc,self.inc,self.argp,self.lan,self.Meananom,self.kep)
        self.pos = pos
        self.vel = vel
        self.acc = acc
        # Convert au to mas:
        self.dec_mas = (pos[:,0].value / self.distance)*1000
        self.ra_mas = ((pos[:,1].value / self.distance)*1000) * np.cos(np.radians(planet.dec))
        # compute separation:
        self.sep_mas = np.sqrt(self.ra_mas**2 + self.dec_mas**2)
        # compute phase angle for each point:
        phases = []
        for i in range(len(self.sma)):
            phase = GetPhaseAngle(self.Meananom[i], self.ecc[i], self.inc[i], self.argp[i])
            phases.append(phase)
        self.phases = phases

        
def MakeCloudPlot(planet, points, lim = 50, plot_contours = True):
    ''' For an OrbitSim object, make a plot of the array of points at a specific date.

    args:
        points [OrbitSim object]: OrbitSim object created from a Planet object
        lim [int]: limit of plot axis in mas
        plot_contours [bool]: if True, plot 1, 2, and 3 sigma contour lines. Default = True
    '''
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    plt.scatter(0,0, marker='*',color='orange',s=300, zorder=10)
    linestyles=[':','--','-']
    pp = ax.scatter(points.ra_mas,points.dec_mas, ls='None', marker='.', alpha = 0.7, c=points.phases, cmap='viridis', s=0.1)
    if plot_contours:
        H, xbins, ybins, midpoints, clevels = GetHist(points.ra_mas, points.dec_mas, sigmas = [1,2,3])
        CS1 = ax.contour(*midpoints, gaussian_filter(H.T, sigma=2), levels = clevels, 
                        linewidths=3, linestyles = linestyles, colors=['orange']*len(linestyles))
    cbar = plt.colorbar(pp)
    cbar.ax.set_ylabel('Viewing Phase [deg]')
    ax.set_aspect('equal')
    ax.set_xlim(-lim*np.cos(np.radians(planet.dec)),lim*np.cos(np.radians(planet.dec)))
    ax.set_ylim(-lim,lim)
    ax.invert_xaxis()
    ax.set_xlabel('$\Delta$RA$^{*}$ [mas]')
    ax.set_ylabel('$\Delta$DEC [mas]')
    ax.grid(ls=':')
    return fig


def MakeKDEPlot(points, lim = 50, kdesize = 50j, plot_contours = True, sigmas = [1,2,3]):
    ''' For an OrbitSim object, make a plot of the probability density of points at a specific date.

    args:
        points [OrbitSim object]: OrbitSim object created from a Planet object
        lim [int]: limit of plot axis in mas
        kdesize [complex int]: Input to np.mgrid for KDE size
    '''
    import matplotlib.pyplot as plt
    ind = np.isfinite(points.dec_mas)

    kde, xmin, ymin, xmax, ymax = GetKDE(points.ra_mas[ind],points.dec_mas[ind], size=kdesize)
    kdenormed = kde/np.sum(kde) # sums to 1 -> it's a PDF

    fig,ax = plt.subplots()
    plt.scatter(0,0, marker='*',color='orange',s=300, zorder=10)
    a = ax.imshow(kdenormed, cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax], origin='lower')
    cbar = plt.colorbar(a)
    cbar.ax.set_ylabel('Probability Density')
    if plot_contours:
        linestyles=[':','--','-']
        clevels = GetCLevels(kdenormed, sigmas = sigmas)
        xe = np.linspace(xmin,xmax,kde.shape[0]+1)
        ye = np.linspace(ymin,ymax,kde.shape[0]+1)
        midpoints = (xe[1:] + xe[:-1])/2, (ye[1:] + ye[:-1])/2
        CS1 = ax.contour(*midpoints, gaussian_filter(kdenormed, sigma=1), levels = clevels, 
                      linewidths=3, linestyles = linestyles, colors=['orange']*len(linestyles))

    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.invert_xaxis()
    ax.set_aspect('equal')
    ax.set_xlabel('$\Delta$RA$^{*}$ [mas]')
    ax.set_ylabel('$\Delta$DEC [mas]')
    ax.grid(ls=':')
    return fig


def GetGaiaOrbitalElementsWithNSSTools(GaiaDR3_sourceid):
    ''' For solutions in Gaia DR3 Non-single star catalog (NSS) with solution type "Orbital",
    query the NSS catalog, retrieve Thiele Innes elements, anf convert them to Campbell elements.

    Args:
        GaiaDR3_sourceid (str): Gaia DR3 source id

    Returns:
        dict: Campbell elements
    '''
    from astroquery.gaia import Gaia
    from nsstools import NssSource
    st = ''' select * from gaiadr3.nss_two_body_orbit where source_id = 
        '''+str(GaiaDR3_sourceid)
    j = Gaia.launch_job(st)
    r = j.get_results()
    r = r.to_pandas()
    r['source_id'] = r['SOURCE_ID']
    source = NssSource(r, indice=0)
    camp = source.campbell()

    T0 = 2016.0 + (r['t_periastron'][0]*u.d.to(u.yr)) # # days since 2016.0
    elements = {"sma [mas]":[camp['a0'][0],camp['a0_error'][0]],
                'ecc':[r['eccentricity'][0], r['eccentricity_error'][0]],
                'inc [deg]':[camp['inclination'][0],camp['inclination_error'][0]],
                'argp [deg]': [camp['arg_periastron'][0],camp['arg_periastron_error'][0]],
                'lan [deg]': [camp['nodeangle'][0],camp['nodeangle_error'][0]],
                'T0 [yr]': [T0,r['t_periastron_error'][0]*u.d.to(u.yr)],
                'P [d]': [r['period'][0], r['period_error'][0]],
                'Chi2':r['obj_func'][0],
                'GOF':r['goodness_of_fit'][0],
                'Efficiency': r['efficiency'][0],
                'Significance':r['significance'][0]
               }
    return elements



def GetFracWithinAperture(planet, sigma_t0, sigma_argp, aperture_size):
    argp = [planet.argp[0], sigma_argp]
    t0 = [planet.t0[0], sigma_t0*u.d.to(u.yr)]
    planet.t0 = t0
    planet.argp = argp
    points = OrbitSim(planet, planet.date_of_max_elongation)
    sep = np.sqrt(
        (points.ra_mas - planet.ras_mean_params[planet.ind_of_max_elongation[0]])**2 +
        (points.dec_mas - planet.decs_mean_params[planet.ind_of_max_elongation[0]])**2
        )
    if type(aperture_size) == float or type(aperture_size) == int:
        stats = np.where(sep < aperture_size)[0].shape[0]/sep.shape[0]
    else:
        stats = []
        for ap in aperture_size:
            stats.append(np.where(sep < ap)[0].shape[0]/sep.shape[0])
    return stats

def QuantifyErrorImprovement(planet, T0_err_array, argp_err_array, aperture_size):
    if type(aperture_size) == float or type(aperture_size) == int:
        probarrs = np.empty((1, T0_err_array.shape[0], argp_err_array.shape[0])) * np.nan
    else:
        probarrs = np.empty((len(aperture_size), T0_err_array.shape[0], argp_err_array.shape[0])) * np.nan

    for i,sigt in enumerate(T0_err_array):
        for j,sigar in enumerate(argp_err_array):
            g = GetFracWithinAperture(planet, sigt, sigar, aperture_size)
            for p in range(probarrs.shape[0]):
                probarrs[p,i,j] = g[p]
        update_progress(i+1,T0_err_array.shape[0])
    return probarrs

def GetContours(arr, levels = [0.16,0.5,0.84]):
    Hflat = arr.T.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except IndexError:
            V[i] = Hflat[0]
    clevels = np.sort(V)
    return clevels
