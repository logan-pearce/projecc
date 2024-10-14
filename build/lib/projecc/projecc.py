import numpy as np
import astropy.units as u
import astropy.constants as c

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
        vel[0], vel[1] = (( -meanmotion * sma * np.sin(E) ) / ( 1- ecc * np.cos(E) )).to(u.km/u.s).value, \
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
    phi = ((np.degrees(np.arctan2(-pos[1].value,pos[0].value))) ) % 360

    return r, phi

def GetSepAndPA(pos):
    r = np.sqrt(pos[:,0]**2 + pos[:,1]**2).value
    phi = ((np.degrees(np.arctan2(-pos[:,1].value,pos[:,0].value))) ) % 360
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