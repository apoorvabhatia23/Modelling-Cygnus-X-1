# Importing the essential modules 

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.integrate as integrate
from scipy.integrate import quad
from scipy.special import kn
from scipy.interpolate import make_interp_spline
import math as mt


# Defining some of the constants that would be used throughout the code

c = 3e10                          # Speed of light in vaccum in cm/s
h = 6.67e-27                      # Planck's constant in erg/s
G = 6.67e-8                       # Gravitational constant in erg cm/g^2
m_e = 9.1e-28                     # Electron mass in g
e = 5e-10                         # Electron charge in esu
D = 1.86e3 * 3.0e18               # Distance of Cyg X-1 from Earth in cm
M = 14.8*2e33                     # Mass of Cyg X-1 BH in g
r_g = 2*G*M/(c**2)                # Schwarzschild radius of the BH
R = 20*r_g                        # Radius of corona in cm (~ 8.77e7 cm)
k = 1.4e-16                       # Boltzmann constant in erg/K 
sigma_t = 6.65e-25                # Thomson cross-section in cm^2   
T = 1e9                           # Temperature of the corona in K
s = 2.4                           # Particle distribution index (data shows it should be between 2.3 - 2.5)
L = 4.5e37                        # Luminosity of Cyg X-1 in erg/s
m_dot = 8e17                      # Mass loss rate in g/s
eta = 1/16                        # Efficiency
B = 1e5                           # Magnetic field in G
pitch = np.pi/2                   # Pitch angle


# Some other parameters required for calculations

V = 4*np.pi*R**3/3                # Volume lobe

L_edd = m_dot*(c**2)/0.5          # Eddington Luminsoity of Cyg X-1 in erg/s                     

gamma_min = 1.3                   # Minimum Lorentz factor
gamma_max = 1e3                   # Maximum Lorentz factor

# Frequency of gyration (talking about angular frequency)

wb_min = e*B/(gamma_min*m_e*c)
wb_max = e*B/(gamma_max*m_e*c)

# Critical frequency (talking about angular frequency)

wc_min = 3*(gamma_min**3)*wb_min*np.sin(pitch)/2
wc_max = 3*(gamma_max**3)*wb_min*np.sin(pitch)/2

#Varying the angular frequency

omega = np.logspace(0, np.log10(wc_max), 100000)
nu = omega/(2*np.pi)


# Determining the direction of the photon from the accretion disk

def random_direction(number = None):
    
    if number is None:
        number = 1
    
    theta = 2.0*np.pi*np.random.rand(number)
    cos_phi = 2.0*np.random.rand(number) - np.ones(number)
    sin_phi = np.sqrt(1 - cos_phi**2)
    
    return((np.array([np.cos(theta)*sin_phi, np.sin(theta)*sin_phi, cos_phi])).transpose())


# Returns the seed photon energy

def f_of_hnu_planck(mc_parms, number = None, pdf = None, energies = None):
    if number is None:
        number = 1
    if (pdf is None):
        e, pdf, energies = hnu_of_p_planck(number = number)
    else:
        e, pdf, energies = hnu_of_p_planckl(number = number, pdf = pdf, hnu = energies)        
    e *= mc_parms['kt_seeds']    
    return(e)


# Determining the origin of the photons in the disk (assuming they all originate from one point)

def photon_origin(number = None):
    
    if number is None:
        number = 1
    
    return(np.zeros([number, 3]))


# Drawing a seed with energy determined by the Planck Function (assuming the disk is a blackbody)

def draw_seed_photons(mc_parms, number = None):
    
    if number is None:
        number = 1
    
    x_seed = photon_origin(number = number)
    n_seed = random_direction(number = number)
    hnu = mc_parms['hnu_dist'](mc_parms, number = number)
    p_seed = (np.array([hnu, hnu*n_seed[:,0], hnu*n_seed[:,1], hnu*np.abs(n_seed[:,2])])).transpose()/c
    
    return(p_seed, x_seed)


# Probability of the seed (of photon) to get scattered in the corona

def tau_of_scatter():

    tau = -np.log(np.random.rand())
    return(tau)


# Converting Optical Depth to Path Length

def distance_of_scatter(mc_parms):

    tau = tau_of_scatter()
    electron_density = mc_parms['tau']/mc_parms['H']/sigma_t
    distance = tau/sigma_t/electron_density
    
    return(distance)


# Getting the scattering Location

def scatter_location(x_old,p_photon,mc_parms):
    
    # ...path-length:
    distance = distance_of_scatter(mc_parms)
    
    # ...in direction:
    photon_direction = p_photon[1:]/p_photon[0]
    
    # Update photon position with the new location
    x_new = x_old + distance*photon_direction
    
    return(x_new)


# Momentum Distribution for electrons (using Maxwell-Jüttner Distribution)

def analytic_juttner(mc_parms):
    #list of large amount of beta values
    beta_list = np.linspace(0, 1.0, 10000)

    #calculate values of gamma as function of momentum for all beta's
    velocity = beta_list* c
    gamma_v = np.sqrt(1/ (1-(beta_list)**2))
    momentum =  gamma_v * m_e * velocity
    gamma_p = np.sqrt(1 + (momentum/(m_e * c))**2)

    ratio = mc_parms['kt_electron']/(m_e * (c ** 2))

    #Maxwell-Juttner Distribution in terms of momentum
    f_p = (1/(4 * np.pi * (m_e**3) * (c**3) * ratio * kn(1, 1/ratio)) * np.exp(- gamma_p/ratio))
    
    
    #calculate an analytic CDF for the Maxwell-Juttner distribution
    pdf_juttner = []
    for i in range(len(f_p)):
        pdf_juttner.append(sum(f_p[0:i]))

    cdf_juttner = np.array(pdf_juttner)/pdf_juttner[-1]

    return gamma_p, cdf_juttner

def gamma_to_velocity(gamma):
    
    #quick transform from gamma to velocity in cm/s
    beta = np.sqrt(1 - (1/gamma**2))
    velocity = (beta)*c
    return velocity

def velocity_to_gamma(velocity):
    gamma = 1/(np.sqrt(1-(velocity/c)**2))
    return gamma

def f_of_v_maxwell_juttner(mc_parms):    
    
    #This is the bread and butter, here we use a random number generator to sample the Juttner CDF
    gamma_p = mc_parms['gamma_p']
    cdf_juttner = mc_parms['cdf_juttner']
    random_number = np.random.uniform(low=mc_parms['min_prob'], high=1.0, size=1)
    juttner_gamma_p = np.interp(random_number, cdf_juttner, gamma_p)
    
    velocity = gamma_to_velocity(juttner_gamma_p)
    
    return velocity[0] #velocity is a list made up of one element


# Drawing random electron velocities and their directions

def draw_electron_velocity(mc_parms, p_photon):
    """Returns a randomized electron velocity vector for inverse 
       Compton scattering, taking relativistic foreshortening of the
       photon flux in the electron frame into account
       
       Args:
           mc_parms (dictionary): Monte-Carlo parameters
           p_photon (4 dimentional np array): Photon 4-momentum
           
       Returns:
           3-element numpy array: electron velocity
    """
    v = mc_parms['v_dist'](mc_parms)
    n = draw_electron_direction(v, p_photon)
    return(v*n)

def draw_electron_direction(v, p_photon):
    """Draw a randomized electron direction, taking account of the
       increase in photons flux from the foward direction, which
       effectively increases the cross section for forward scattering.
       
       Args:
            v (real): electron speed
            p_photon (4 element numpy array): photon 4-momentum
            
       Returns:
           3-element numpy array: randomized electron velocity vector
    """
    phi = 2.*np.pi*np.random.rand()
    cosp = np.cos(phi)
    sinp = np.sin(phi)
    cost = mu_of_p_electron(v/c,np.random.rand())
    sint = np.sqrt(1 - cost**2)
    
    n_1 = p_photon[1:]/p_photon[0]
    if (np.sum(np.abs(n_1[1:2])) != 0):
        n_2 = np.cross(n_1,np.array([1,0,0]))
    else:
        n_2 = np.cross(n_1,np.array([0,1,0]))
    n_2 /= np.sqrt(np.sum(n_2**2))
    n_3 = np.cross(n_1,n_2)
    
    # express new vector in old base
    n_new = (n_2*cosp+n_3*sinp)*sint + n_1*cost
    return(n_new/np.sqrt(np.sum(n_new**2)))


# Getting the foreshortened Thomson scattering cross-section

def mu_of_p_electron(beta,p):
    """Invert probability for foreshortened effective
       Thomson scattering cross section, with
    
       P = 
       
       Args:
           beta (real): v/c for electron
           p: probability value between 0 and 1
           
       Returns:
           real: cos(theta) relative to photon direction
    """
    mu = 1/beta-np.sqrt(1/beta**2 + 1 - 4*p/beta + 2/beta)
    return(mu)


# Lorentz Transform

def lorentz_transform(p,v):
    """Returns general Lorentz transform

    Args:
        p (four-element numpy array): input four-vector
        v (three-element numpy array): the 3-velocity of the frame we want to transform into

    Returns:
        four-element numpy array: the transformed four-vector
    """

    beta = np.sqrt(np.sum(v**2))/c
    beta_vec = v/c
    gamma = 1./np.sqrt(1. - beta**2)
    matrix = np.zeros((4,4))
    matrix[0,0] = gamma
    matrix[1:,0] = -gamma*beta_vec
    matrix[0,1:] = -gamma*beta_vec
    matrix[1:,1:] = (gamma-1)*np.outer(beta_vec,beta_vec)/beta**2
    for i in range(1,4):
        matrix[i,i] += 1
    return(np.dot(matrix,p))


# Definig the angle of scattering

def cos_theta_thomson(p):

    a = -4 + 8*p
    b = a**2 + 4
    
    return((np.power(2,1/3)*np.power(np.sqrt(b)-a,2/3)-2)/
           (np.power(2,2/3)*np.power(np.sqrt(b)-a,1/3)))


# Thomson Scattering Machinery

def thomson_scatter(p_photon):
    """This function performs Thomson scattering on a photon
    
    Args:
        p_photon (4-element numpy array): Incoming photon four-vector
        
    Returns:
        4-element numpy array: Scattered photon four-vector
    """
    
    n_1 = p_photon[1:]/p_photon[0]
    if (np.sum(np.abs(n_1[1:2])) != 0):
        n_2 = np.cross(n_1,np.array([1,0,0]))
    else:
        n_2 = np.cross(n_1,np.array([0,1,0]))
    n_2 /= np.sqrt(np.sum(n_2**2))
    n_3 = np.cross(n_1,n_2)

    # scattering is uniform in phi
    phi = 2.*np.pi*np.random.rand()
    cosp = np.cos(phi)
    sinp = np.sin(phi)
    
    # draw cos_theta from proper distribution
    cost = cos_theta_thomson(np.random.rand())
    sint = np.sqrt(1 - cost**2)
    
    # express new vector in old base
    n_new = (n_2*cosp+n_3*sinp)*sint + n_1*cost
    n_new /= np.sqrt(np.sum(n_new**2))
    
    # return scatterd 4-momentum vector
    return(np.array(p_photon[0]*np.array([1,n_new[0],n_new[1],n_new[2]])))


# Inverse Compton Scattering

def inverse_compton_scatter(p_photon,mc_parms):
    
    # throw the dice one more time to draw a random electron velocity
    velocity = draw_electron_velocity(mc_parms,p_photon)
    # first, transform to electron frame
    p_photon_prime = lorentz_transform(p_photon,velocity)

    # Thomson scatter
    p_out_prime = thomson_scatter(p_photon_prime)
    
    # transform back to observer frame
    return(lorentz_transform(p_out_prime, -velocity))


# Monte-Carlo Function

def monte_carlo(mc_parms):
    
    # arrays to store initial and final photon energies
    hnu_seed = np.zeros(mc_parms['n_photons'])
    hnu_scattered = hnu_seed.copy()

    # draw our seed-photon population. Much faster to do this once for all photons
    p_photons, x_photons = draw_seed_photons(mc_parms, number=mc_parms['n_photons'])
   
    # run the scattering code n_photons times
    for p_photon, x_photon, i in zip(p_photons, x_photons, range(mc_parms['n_photons'])):
        # initial photon four-momentum
        # store seed photon energy for future use (calculating Compton-y parameter)
        hnu_seed[i] = p_photon[0]*c

        # keep scattering until absorbed or escaped
        scattered = True
        while (scattered):
            # find next scattering location
            x_photon = scatter_location(x_photon, p_photon, mc_parms)
            # if it's inside the corona, perform inverse Compton scatter
            if (x_photon[2] >= 0 and x_photon[2] <= mc_parms['H']):
                p_photon = inverse_compton_scatter(p_photon, mc_parms)
            else:
                scattered=False
                if (x_photon[2] <= 0):
                    p_photon *= 0

        # store the outgoing photon energy in the array
        hnu_scattered[i] = p_photon[0]*c

    # only return escaped photons and their seed energy
    return(hnu_scattered[hnu_scattered > 0], hnu_seed[hnu_scattered > 0])


# Plotting the distribution (for inverse compton)

def plot_mc(mc_parms,bins=None,xlims=None,filename=None):
    
    #global figure_counter
    
    # Now run simulation and normalize all outgoing photon energies 
    # so we can investigate energy gains and losses
    hnu_scattered, hnu_seeds = np.array(monte_carlo(mc_parms))/mc_parms['kt_seeds'] 
    
    if (xlims is None):
        xlims = [hnu_scattered.min(), hnu_scattered.max()]    
    if (bins is None):
        bins = np.logspace(np.log10(xlims[0]), np.log10(xlims[1]), num=100)
    else:
        bins = np.logspace(np.log10(xlims[0]), np.log10(xlims[1]), num=bins)

    print('Fraction of escaping photons: {0:5.3e}\n'.format(hnu_scattered.size/mc_parms['n_photons']))
    print('Compton y parameter: {0:5.3e}\n'.format(compton_y(hnu_seeds,hnu_scattered)))
    return(hnu_scattered, hnu_seeds)


# Photon distribution function of a Planck distribution

def f_planck(x):
    """Photon distribution function of a Planck distribution
    
    Args:
        x (real): photon energy e in untis of kT
        
    Returns:
        real: differential photon number dN/de at energy e
    """
    norm = 2.4041138063192817
    return x**2/(np.exp(x) - 1)/norm

def p_planck(hnu=None):
    """Numerical integration of cumulative planck PDF (to be inverted)
    
    Parameters:
        hnu (numpy array): bins of photon energy e in untis of kT
        
    Returns:
        numpy array: cumulative photon PDF as function of hnu
        numpy array: hnu values used for PDF
    """
    if (hnu is None):
        number = 1000
        hnu = np.append(np.linspace(0, 1 - 1./number, number),np.logspace(0, 4, number))

    p = np.zeros(2*number)
    for i in range(1,2*number):
        p[i] = ((quad(f_planck, 0, hnu[i]))[0])
    return (p,hnu)

def hnu_of_p_planck(number = None, pdf = None, hnu = None):
    if number is None:
        number = 1
    if (pdf is None):
        pdf, hnu = p_planck()

    e_phot = np.interp(np.random.rand(number), pdf, hnu)
    return(e_phot, pdf, hnu)

def f_of_hnu_planck(mc_parms, number = None, pdf = None, energies = None):
    if number is None:
        number = 1
    if (pdf is None):
        e, pdf, energies = hnu_of_p_planck(number = number)
    else:
        e, pdf, energies = hnu_of_p_planckl(number = number, pdf = pdf, hnu = energies)        
    e *= mc_parms['kt_seeds']   
    return(e)


# Defining a fucntion as to get the energy of electrons from the non-thermal component (esentially the SSA)

def f_of_hnu_nonthermal(mc_parms):
    

    v_list = mc_parms['v_list']
    e_energy_dist = np.sqrt(m_e**2 * c**4 + (velocity_to_gamma(np.array(v_list))*m_e*np.array(v_list))**2 * c**2)

    h = scipy.constants.Planck*1e7 # erg

    N_E_electrons, energy_bins =  np.histogram(e_energy_dist, bins=10)
    mean_energy_bins = running_mean_convolve(energy_bins, 2)

    E_tot = N_E_electrons * mean_energy_bins

    nu_list = E_tot/h

    random_energy = 0.0
    while random_energy == 0.0:
        random_energy = E_tot[np.random.randint(0, len(E_tot))]

    e = random_energy

    return np.array([e])   


# To calculate the Compton-Y paramter

def compton_y(pre,post):
    y = np.mean((post-pre)/pre)
    return(y)


# To convert the histogram (from the distribution) to give a proper flux vs frequency graph

def running_mean_convolve(x, N):
    return np.convolve(x, np.ones(N) / float(N), 'valid')

def num_to_flux(hnu_scattered):
    h = 6.6e-27
    N_hnu, bin_array = np.histogram(hnu_scattered, bins=100)
    avg_bin_array = running_mean_convolve(bin_array, 2) * mc_parms['kt_seeds'] * (1/h)
    flux = (N_hnu * h * avg_bin_array)#/ (2*np.pi * mc_parms['H']**2)

    plt.figure(figsize=(16, 9))
    plt.plot(avg_bin_array , flux, label=r'$\tau$ = {}'.format(mc_parms['tau']))
    plt.xlabel(r'$\nu$', fontsize=20)
    plt.ylabel(r'$F_{\nu}$', fontsize=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.show()
   
    return flux, avg_bin_array


# Spectrums

# First mc_parms to for analytic_juttner to use

temp_mc_parms={'n_photons':100000,              # assuming only a fraction of photons from the disk interact with the corona
          'kt_seeds':0.3 * 1.6e-9,              # 0.3 keV input photons
          'H':R,                                # say H ~ R, and R ~ 20 R_g ~ 8.77e7 cm
          'tau':0.3,                            # Optical Depth of corona
          'kt_electron':100 * 1.6e-9,           # 100 keV (For corona, it should be between 50-100 keV)
          'v_dist':f_of_v_maxwell_juttner,      # name of electron velocity distribution function
          'hnu_dist':f_of_hnu_planck,           # name of photon distribution function
          'return_v':False,                     # condition to return the velocity of the electron
         }


gamma_p, cdf_juttner = analytic_juttner(temp_mc_parms)


# Second mc_parms which includes the outputs of analytic_juttner

mc_parms={'n_photons':100000,                   # assuming only a fraction of photons from the disk interact with the corona
          'kt_seeds':0.3 * 1.6e-9,              # 0.3 keV input photons
          'H':R,                                # say H ~ R, and R ~ 20 R_g ~ 8.77e7 cm
          'tau':0.3,                            # Optical Depth of corona
          'kt_electron':75 * 1.6e-9,            # 100 keV (For corona, it should be between 50-100 keV)
          'v_dist':f_of_v_maxwell_juttner,      # name of electron velocity distribution function
          'hnu_dist':f_of_hnu_planck,           # name of photon distribution function
          'return_v':False,                     # condition to return the velocity of the electron
          'gamma_p':gamma_p,
          'cdf_juttner':cdf_juttner,
          'min_prob':0.0,
         }


f_of_hnu_planck(mc_parms)


# Defining the accretion disk as a blackbody

def blackbody_disk():
    
    B_nu = ((2*h*(nu**3))/(c**2))*(1/(np.exp(h*nu/(mc_parms['kt_seeds'])) - 1))

    # Flux on surface
    F_nu_bb = 4*np.pi*B_nu

    # Flux on Earth
    F_nu_earth = F_nu_bb*((R/D)**2)              
    plt.figure(figsize=(16, 9))    
    return F_nu_bb


# Defining the inverse compton model

def IC():
    hnu_scattered_IC, hnu_seeds = plot_mc(mc_parms)
    flux, avg_bin_array = num_to_flux(hnu_scattered_IC)
    return avg_bin_array, flux


# Defining the self-synchrotron absorption mode

def SSA():
    I_nu_arr = []
    F_nu_arr_earth = []
    F_nu_arr = []
    v_list = []
    mc_parms_ssa={'n_photons':100000,         # assuming only a fraction of photons from the disk interact with the corona
          'kt_seeds':0.3*1.6e-9,              # 0.3 keV input photons
          'H':R,                              # say H ~ R, and R ~ 20 R_g ~ 8.77e7 cm
          'tau':0.3,                          # Optical Depth of corona
          'kt_electron':100*1.6e-9,           # 100 keV (For corona, it should be between 50-100 keV)
          'v_dist':f_of_v_maxwell_juttner,    # name of electron velocity distribution function
          'hnu_dist':f_of_hnu_planck,         # name of photon distribution function
          'return_v':True,                    # condition to return the velocity of the electron
          'gamma_p':gamma_p,
          'cdf_juttner':cdf_juttner,
          'min_prob':0.0,
         }

    for i in range(100000):
        v = f_of_v_maxwell_juttner(mc_parms_ssa)
        
        if v >= 0.7*c and v < c:
            v_list.append(v)
            C = (eta*L_edd)/(4*np.pi*(R**2)*v)*(1/np.log(gamma_max))  
            alpha_nu = (np.sqrt(3)*e**3)/(8*np.pi*m_e)*np.power((3*e)/(2*np.pi*m_e**3*c**5), s/2)*C*np.power(B*np.sin(pitch), (s+2)/2)*mt.gamma((3*s+2)/12)*mt.gamma((3*s+22)/12)*np.power(omega, -(s+4)/2)

            # Total power per unit volume per unit frequency (6.36 R&L)

            P_tot_omega = (np.sqrt(3)*e**3*C*B*np.sin(pitch))/(2*np.pi*m_e*c**2*(s+1))*mt.gamma(s/4+19/12)*mt.gamma(s/4-1/12)*np.power((m_e*c*omega)/(3*e*B*np.sin(pitch)), -(s-1)/2)
            P_tot_nu = 2*np.pi*P_tot_omega

            # Source function

            S_nu = P_tot_nu/(4*np.pi*alpha_nu)

            # Optical depth

            tau_nu = alpha_nu*R

            # Specific Intensity

            I_nu = S_nu*(1-np.exp(-tau_nu))        # erg/s/cm^2/Hz/ster

            I_nu_arr.append(I_nu)

            F_nu = I_nu * 4 * np.pi
            
            F_nu_arr.append(F_nu)

#             F_nu_earth = F_nu * ((R/D)**2)

#             F_nu_arr_earth.append(F_nu_earth)

#             I_nu_tot = sum(I_nu)
#             F_nu_earth_tot = sum(F_nu_arr_earth)

#             L_nu_tot = F_nu_earth_tot * (4*np.pi*D**2)
        else:
            continue
        F_nu_tot = sum(F_nu_arr)

    print(len(v_list))
    return(nu, F_nu_arr, F_nu_tot, v_list)


# Defining the self synchrotron compton model

def SSC(v_list):
    mc_parms_ssc={'n_photons':len(v_list),         # assuming only a fraction of photons from the disk interact with the corona
      'kt_seeds':np.sqrt(((m_e**2)*(c**4)) + ((1/np.sqrt(1 - (np.mean(v_list)/c)**2))*m_e*np.mean(v_list)*c)**2),              # relativistic electrons are our input seeds here
      'H':R,                              # say H ~ R, and R ~ 20 R_g ~ 8.77e7 cm
      'tau':0.3,                          # Optical Depth of corona
      'kt_electron':100*1.6e-9,           # 100 keV (For corona, it should be between 50-100 keV)
      'v_dist':f_of_v_maxwell_juttner,    # name of electron velocity distribution function
      'hnu_dist':f_of_hnu_planck,         # name of photon distribution function
      'return_v':False,                   # condition to return the velocity of the electron
      'gamma_p':gamma_p,
      'cdf_juttner':cdf_juttner,
      'min_prob':0.9,
      'v_list':v_list,
                 }
    
    hnu_scattered_ssc, hnu_seeds_ssc = plot_mc(mc_parms_ssc)
    flux_ssc, avg_bin_array_ssc = num_to_flux(hnu_scattered_ssc)

    return flux_ssc, avg_bin_array_ssc


# Calling the blackbody_disk function

F_nu_bb = blackbody_disk()

# Calling the inverse compton function

nu_ic, flux_IC = IC()

# Calling the self-synchrotron absorption function

nu, F_nu_arr, F_nu_tot, v_list = SSA()

# Calling the self-synchrtron comptom function

flux_ssc, nu_ssc = SSC(v_list)


# Plotting the electron energy distribution and photon distribution for SSA

e_energy_dist = np.sqrt(m_e**2 * c**4 + (velocity_to_gamma(np.array(v_list))*m_e*np.array(v_list))**2 * c**2)
plt.figure(figsize=(16, 9))

plt.hist(e_energy_dist, bins=100)
plt.xlabel('Electron energy [Erg]')
plt.ylabel(r'$N(E_{e^-})$')
plt.title('Electron energy distribution')
plt.show()

plt.scatter(np.array(v_list)/c, e_energy_dist)
plt.axvline(1)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$E_{e^-}$ [Erg]')
plt.xscale('log')
plt.show()

h = scipy.constants.Planck*1e7 # erg

N_E_electrons, energy_bins =  np.histogram(e_energy_dist, bins=100)
mean_energy_bins = running_mean_convolve(energy_bins, 2)

E_tot = N_E_electrons * mean_energy_bins

nu_list = E_tot/h


plt.figure(figsize=(16, 9))

plt.scatter(mean_energy_bins/h, E_tot)
plt.xlabel(r'$\nu[Hz]$')
plt.ylabel('Energy [erg]')
# plt.xscale('log')
# plt.yscale('log')
plt.title('Photon energy distribution for SSA')
plt.show()


# Plotting the total spectrum

plt.figure(figsize=(16, 9))
plt.plot(nu, F_nu_tot, label="SSA")
plt.plot(nu, F_nu_bb, label='BB')
plt.scatter(nu_ic, flux_IC, label='IC', s=5, color='red')
plt.scatter(nu_ssc, flux_ssc, label='SSC', s=5, color='green')
plt.scatter(mean_energy_bins/h, E_tot, label='input SSA photon field', s=5, color='purple')
plt.xscale('log')
plt.yscale('log')
plt.title('Total spectrum')
plt.xlabel(r'$\nu$[Hz]')
plt.ylabel(r'$F_{\nu}[erg * s^{-1} * cm^{-2}]$')
plt.ylim(1e-36, 1e6)
plt.legend(loc='upper left')
plt.show()    


x1, y1 = [1,2,3],[1,1,1]
x2, y2 = [1.5,2.5],[2,2]
# get a sorted list of all x values
x = np.unique(np.concatenate((nu, nu, nu_ic, nu_ssc, mean_energy_bins/h)))
# interpolate y1 and y2 on the combined x values
yi1 = np.interp(x, nu, F_nu_tot, left=0, right=0)
yi2 = np.interp(x, nu, F_nu_bb, left=0, right=0)
yi3 = np.interp(x, nu_ic, flux_IC, left=0, right=0)
yi4 = np.interp(x, nu_ssc, flux_ssc, left=0, right=0)
yi5 = np.interp(x, mean_energy_bins/h, E_tot, left=0, right=0)


plt.figure(figsize=(16, 9))
plt.scatter(nu, F_nu_tot, label="SSA", s=1)
plt.scatter(nu, F_nu_bb, label='BB', s=1)
plt.scatter(nu_ic, flux_IC, label='IC', s=1, color='red')
plt.scatter(nu_ssc, flux_ssc, label='SSC', s=1, color='green')
plt.scatter(mean_energy_bins/h, E_tot, label='input SSA photon field', s=100, color='purple')
plt.scatter(x, yi1 + yi2 + yi3 + yi4 +yi5, label="Total spectrum", s=5)
plt.xscale('log')
plt.yscale('log')
plt.title('Total spectrum')
plt.xlabel(r'$\nu$[Hz]')
plt.ylabel(r'$F_{\nu}[erg * s^{-1} * cm^{-2}]$')
plt.ylim(1e-36, 1e6)
plt.legend(loc="upper left")
plt.show()


# Plotting the CDF of Maxwell-Juttner Distribution

plt.figure(figsize=(16, 9))
plt.plot(gamma_p, cdf_juttner)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\gamma_p$')
plt.ylabel(r'P($\gamma_p$)')
plt.title('CDF of Maxwell-Juttner distribution')
plt.show()


# Plotting the distribution for the disk

plt.figure(figsize=(16, 9))
plt.loglog(nu, F_nu_bb, label='BB')
plt.ylim(1e-36, 1e6)
plt.xlabel(r'$\nu$[Hz]')
plt.ylabel(r'$F_{\nu}[erg * s^{-1} * cm^{-2}]$')
plt.title('Planck distribution for disk at 0.3 keV')
plt.show()