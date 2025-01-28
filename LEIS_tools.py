"""
This package can be used to postprocess experimental and simulated LEIS spectra. It allows 
to consider real solid angles of elements, determine peak posistions and cor__responding elements,
calculate intensity correction coefficients, etc. It thus can be useful in LEIS.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

If you have questions regarding this program, please contact NEEfimov@mephi.ru
"""

import numpy as np
import matplotlib.pyplot  as plt    
import scipy.signal
import math, re, os
from mendeleev import get_all_elements


######################    GLOBAL SETTINGS   ############################

step = 2.0 #energy step, eV
Emin = 300.0 #minimal energy, eV
Emax = 30000.0 #maximal energy, eV

######################    GLOBAL SETTINGS   ############################


# the most time consuming procedure, loads all data on the chemical elements
elements = get_all_elements()
print("periodic chemical elements database is LOADED")

class spectrum:
    
    def __init__(self, step: float):
        self.__step = step
    
    @property
    def spectrum_en(self):
        return self.__spectrum_en

    @property
    def spectrum_int(self):
        return self.__spectrum_int
    
    @property
    def calc_name(self):
        return self.__calc_name
           
    @property
    def step(self):
        return self.__step
    
    @property
    def E0(self):
        """
        Initial energy in eV
        """
        return self.__E0
    
    @property
    def incident_atom(self):
        """
        Symbol of the incident atom
        """
        return self.__incident_atom
    
    @property
    def M0(self):
        """
        mass of insident atom in a.m.u.
        """
        return get_mass_by_element(self.__incident_atom)
    
    @property
    def theta(self):
        """
        Scattering angle in degrees
        """
        return self.__theta
    
    @property
    def dTheta(self):
        """
        Spread of scattering angle in degrees
        """
        return self.__dTheta

    def import_data(self, spectrum_path: str, filter_window_length=-1):  
        """
        Method to import energy spectrum from a file located at spectrum_path
        filter windows length is used for smoothing of the spectrum. 
        It has a default value of -1, which means no smoothing. The positive
        value of filter_window_length will apply Savitzky-Golay filter with 
        polynoms of the 3d order. Filter window length has dimension of energy step [eV].
        """

        try:
            spectrum_file = open(spectrum_path).read()
        except:
            print("ERROR during spectrum import. File not found: "+spectrum_path)
            
        spectrum_file = spectrum_file.replace('\t'," ").replace(",", ".").replace("E","e")
        self.__calc_name = spectrum_path.split(os.sep)[-1].split(".dat")[0]

        lines = spectrum_file.splitlines()
        
        # find letter strings and save its indexes
        __indexes_letter_strings = []
        for i in range(0, len(lines)):
            if  any(c.isalpha() and not ("e" in c) for c in lines[i]) or ("*" in lines[i]) or lines[i]=='':
                __indexes_letter_strings.append(lines[i])
        
        # get initial params from the filename
        self.__incident_atom = re.sub(r'\d', '', spectrum_path.split(os.sep)[-1].split("_")[1].split("keV")[0]).replace(".","")
        self.__E0 = float(spectrum_path.split(os.sep)[-1].split("_")[1].split("keV")[0].split(self.__incident_atom)[1])*1000
        self.__theta = float(spectrum_path.split(os.sep)[-1].split("_")[1].split("keV")[1].split("deg")[0])
        try:
            self.__dTheta = float(spectrum_path.split(os.sep)[-1].split("_")[1].split("keV")[1].split("deg")[1])
        except:
            print("ERROR during spectrum import. Can't find dTheta in the filename")
            self.__dTheta = 1.0
            
        # remove letter strings from lines
        for i in __indexes_letter_strings:
            lines.remove(i)
            
        # create data arrays
        raw_spectrum_int = np.zeros(len(lines))
        raw_spectrum_en = np.zeros(len(lines))

        for i in range(0, len(lines)):
            lines[i] = lines[i]
            data = lines[i].split(" ")
            raw_spectrum_en[i] = float(data[0])
            raw_spectrum_int[i] = float(data[1])

        # do interpolation with new energy step and normalization to 1 in range (Emin, Emax)
        self.__spectrum_en = np.arange(0, raw_spectrum_en[-1], self.__step)
        # scaling range in eV (influence spectra normalization in Web charts and output files)
        self.__spectrum_int = np.interp(self.__spectrum_en,raw_spectrum_en, raw_spectrum_int)
        #global Emax
        # Emax = self.__spectrum_en[-1]-100
        self.__spectrum_int /= max(self.__spectrum_int[int(Emin/step):int(self.__spectrum_en[-1]-100/step)])
        if filter_window_length > 0:
            self._smooth(filter_window_length)
   
    def _smooth(self, filter_window_length: int):
            self.__spectrum_int = scipy.signal.savgol_filter(self.__spectrum_int, int(filter_window_length/self.__step), 3)

    
def norm(signal: str):
    """
    method for spectrum normalization to 1 in range (Emin, Emax)
    """
    return signal/max(signal[int(Emin/step):int(Emax/step)])

def peak(signal: str):
    """
    method for finding maximum of the peak in range (Emin, Emax)
    """
    return max(signal[int(Emin/step):int(Emax/step)])

"""
def set_elements_params():
    global Z0, M0, Z1, M1, Z2, M2, mu1, mu2
    Z0=next((el for el in elements if el.symbol == incident_atom), None).atomic_number #projectile atomic number
    M0=next((el for el in elements if el.symbol == incident_atom), None).atomic_weight #projectile atomic mass
    Z1=next((el for el in elements if el.symbol == target_atom1), None).atomic_number #target atomic number
    M1=next((el for el in elements if el.symbol == target_atom1), None).atomic_weight #target atomic mass
    Z2=next((el for el in elements if el.symbol == target_atom2), None).atomic_number #target atomic number
    M2=next((el for el in elements if el.symbol == target_atom2), None).atomic_weight #target atomic mass
    mu1 = M1/M0
    mu2 = M2/M0
"""
 
def get_angle_by_energy (E0:float, mu:float, E1:float):
    """
    returns angle of scattering for the given energy of the incident particle E0,
    its energy losses E1 and relation mu = M_target/m_incident 
    """
    try:
        # may be its scattered? 
        theta1 = 180/np.pi*np.arccos(0.5*np.sqrt(E1/E0)*(E0/E1*(1-mu)+mu+1))
    except:
        try:
            # may be its recoils? 
            theta1 = 180/np.pi*np.arccos((1+mu)*np.sqrt(E1/(4*E0*mu)))
        except:
            print("Error in get_angle_by_energy") 
    return theta1

"""
def get_angle_by_energy2 (E1, mu):
    try: # scatter
        theta1 = 180/np.pi*np.arccos((M0+M1)*(E1 +E0*(M0**2-M1**2)/(M0+M1)**2)/(2*np.sqrt(E0 * E1)*M0))   
    except: # recoil
        theta1 = (90 - 180 * np.arcsin((M0 + M1) * np.sqrt(E1/(E0*M0*M1))/2)/np.pi)
    return theta1
"""

def get_energy_by_angle(E0:float, mu:float, theta:float):
    """
    returns energy of the incident particle for the given angle of scattering theta,
    initial energy E0 and relation mu = M_target/m_incident
    """
    try:
        # may be its scattered?
        E1 = E0*((np.cos(theta*np.pi/180)+np.sqrt(mu**2-(np.sin(theta*np.pi/180))**2))/(1+mu))**2
    except:
        try:
            # may be its recoils?
            E1 = 4*mu/(1+mu)**2*(np.cos(theta*np.pi/180))**2*E0
        except:
            print("Error in get_energy_by_angle")
    return E1

def get_target_mass_by_energy(theta:float, M0:float, E0:float, E1:float):
    #rel_en = np.sqrt(E1/E0)
    #mu = (2*np.cos(theta*np.pi/180)+rel_en+1/rel_en)/(2*np.cos(theta*np.pi/180)-2*rel_en)
    
    if E1 > E0:
        print("ERROR on get_mass_by_energy: E1 > E0")
        return 0
    if theta > 180:
        print("ERROR on get_mass_by_energy: theta > 180")
        return 0
    if True: #theta >= 90:
        o2 = np.cos(theta * np.pi / 180)
        Ae = E1-E0
        Be = 2*M0*E1-2*o2*np.sqrt(E1*E0)*M0
        Ce = (E1+E0-2*o2*np.sqrt(E1*E0))*M0**2
        
        if (Be**2-4*Ae*Ce<0):
            print("ERROR on get_mass_by_energy: such mu does not exist. Conservation laws violation")
        qe = -0.5*(Be+np.sign(Be)* np.sqrt(Be**2-4*Ae*Ce))
        
        if qe < 0:
            return qe/Ae
        if qe > 0:
            return Ce/qe
    else:
        o2 = np.pi - 2 * (theta * np.pi / 180)
        Ae = np.sqrt(E1/E0)
        Be = -np.sqrt(M0)* np.sin(o2/2)*2;
        Ce = np.sqrt(E1 / E0)*M0
        qe = -0.5*(Be+np.sign(Be)*np.sqrt(Be**2-4*Ae*Ce))
        if qe > 0:
            return (qe / Ae)**2,(Ce / qe)**2
    
    print ("Some error occurred in get_mass_by_energy")
    return 0
    

def get_element_by_mass(mass:float):
    """
    Return Symbol of chemical element with the closest atomic weight to the given mass
    """
    masses = {el.symbol: el.atomic_weight for el in elements if el.atomic_weight}
    
    min_diff = 1000
    element_name = None	
    for name, weight in masses.items():
        diff = abs(weight - mass)
        if diff < min_diff:
            min_diff = diff
            element_name = name
    return element_name


def get_mass_by_element(element_symbol:str):
    """
    returns atomic weight of chemical element by specified symbol 
    """
    return next((el for el in elements if el.symbol == element_symbol), None).atomic_weight 

def get_dSigma(theta:float, mu:float, dE:float):
    """
    method return dOmega of scatted particles in steradians for 
    specific bin size dE of the analyzer at energy position
    cor__responding to mu==M_target/M_incident
    """
    theta10 = get_angle_by_energy(get_energy_by_angle(theta, mu)-dE/2, mu)
    theta11 = get_angle_by_energy(get_energy_by_angle(theta, mu)+dE/2, mu)
    return np.abs(2*np.pi*(np.cos(theta10*np.pi/180)-np.cos(theta11*np.pi/180)))

def get_dBeta(E0:float, theta:float, mu:float, dE:float): 
    """
    method return delta of scattering angle in degrees for 
    specific bin size dE of the analyzer at energy position
    cor__responding to mu==M_target/M_incident
    """
    return  dE/(get_energy_by_angle(E0,theta, mu)*2*np.sin(theta*np.pi/180))*np.sqrt(mu**2-(np.sin(theta*np.pi/180))**2)*180/np.pi

def get_dE(E0:float, theta:float, mu:float, dB:float):
    """
    method return delta of energy in eV for 
    specific detection angle of the analyzer at energy position
    cor__responding to mu==M_target/M_incident
    """
    return  dB*(get_energy_by_angle(E0, theta, mu)*2*np.sin(theta*np.pi/180))/np.sqrt(mu**2-(np.sin(theta*np.pi/180))**2)/180*np.pi

def get_intensity_corrections(E0:float, mu1:float, mu2:float, theta:float, R = -1):
    """
    Calculates intensities correction coefficients for considering
    effect of different collecting angles due to spectra discretization
    To recover true intensities multiply intensity of element with
    peak of highers energy to this coefficient
    
    To take into account real spectrometer geometry with dE/E=const 
    specify any positive R value (R = dE/E).
    """  
    if R <= 0 :
        dBeta2 = get_energy_by_angle(E0, theta, mu2)*2*np.sin(theta*np.pi/180)/np.sqrt(mu2**2-(np.sin(theta*np.pi/180))**2)
        dBeta1 = get_energy_by_angle(E0, theta, mu1)*2*np.sin(theta*np.pi/180)/np.sqrt(mu1**2-(np.sin(theta*np.pi/180))**2)
    if R > 0 :
        dBeta2 = (get_energy_by_angle(E0, theta, mu2))**2*2*np.sin(theta*np.pi/180)/np.sqrt(mu2**2-(np.sin(theta*np.pi/180))**2)
        dBeta1 = (get_energy_by_angle(E0, theta, mu1))**2*2*np.sin(theta*np.pi/180)/np.sqrt(mu1**2-(np.sin(theta*np.pi/180))**2)
    return dBeta2/dBeta1


##########################################      PLOTS     ###################################################

#    /$$$$$$$  /$$        /$$$$$$  /$$$$$$$$ /$$$$$$ 
#   | $$__  $$| $$       /$$__  $$|__  $$__//$$__  $$
#   | $$  \ $$| $$      | $$  \ $$   | $$  | $$  \__/
#   | $$$$$$$/| $$      | $$  | $$   | $$  |  $$$$$$ 
#   | $$____/ | $$      | $$  | $$   | $$   \____  $$
#   | $$      | $$      | $$  | $$   | $$   /$$  \ $$
#   | $$      | $$$$$$$$|  $$$$$$/   | $$  |  $$$$$$/
#   |__/      |________/ \______/    |__/   \______/ 
                                                 
##########################################      PLOTS     ###################################################
                                   
                                                 
def plot_dBeta_map():
    global E0
    E0 = 10000
    step_mu = 0.0005
    min_value_mu = 0.5
    max_value_mu = 20
    number_of_points_mu = int((max_value_mu-min_value_mu)/step_mu)

    step_theta = 0.5 #0.5
    min_value_theta = 10
    max_value_theta = 170
    number_of_points_theta = int((max_value_theta-min_value_theta)/step_theta)
    
    map0 = np.zeros((number_of_points_mu, number_of_points_theta))
    angles = np.zeros(number_of_points_theta)
    mu_values = np.zeros(number_of_points_mu)
    #file = open('leis_out.txt', 'w')
    for i_theta in range (0,number_of_points_theta):
        theta = min_value_theta+i_theta*step_theta
        angles[i_theta] = theta
        min_value_mu = np.sin(theta*np.pi/180)
        for i_mu in range (int(min_value_mu/step_mu)+1,number_of_points_mu):
            mu = min_value_mu+i_mu*step_mu
            mu_values[i_mu] = mu
            map0[i_mu, i_theta] = get_dBeta(theta, mu, 2)/2
            #print(str(map0[i_mu, i_theta])[0:5], end=" ")
           # file.write(str(map0[i_mu, i_theta])[0:7]+" ")
        #file.write("\n")
        #print("\n")
    #file.close()

    #nipy_spectral   gist_ncar
    plt.contourf(angles,mu_values, map0, cmap='gist_ncar', levels=np.linspace(0.001, 0.35, 200))
    plt.text(80, 0.5, '__restricted zone: μ> sin(θ)', fontsize = 13)
    plt.colorbar(label='Δβ/ΔE, degrees/eV', ticks=np.linspace(0.001, 0.35, 10))
    plt.xlabel('scattering angle θ, degrees', fontsize=12)
    plt.ylabel('target atom mass / incident atom mass μ',fontsize=12)
    plt.clim(0.001, 0.35)
    plt.show()

#plot_dBeta_map()

#########################   YOUNG'S EMPIRICAL LEIS SPECTRA APPROXIMATION  #####################################

#    /$$     /$$ /$$$$$$  /$$   /$$ /$$   /$$  /$$$$$$ 
#   |  $$   /$$//$$__  $$| $$  | $$| $$$ | $$ /$$__  $$
#    \  $$ /$$/| $$  \ $$| $$  | $$| $$$$| $$| $$  \__/
#     \  $$$$/ | $$  | $$| $$  | $$| $$ $$ $$| $$ /$$$$
#      \  $$/  | $$  | $$| $$  | $$| $$  $$$$| $$|_  $$
#       | $$   | $$  | $$| $$  | $$| $$\  $$$| $$  \ $$
#       | $$   |  $$$$$$/|  $$$$$$/| $$ \  $$|  $$$$$$/
#       |__/    \______/  \______/ |__/  \__/ \______/ 
                                                   
#########################   YOUNG'S EMPIRICAL LEIS SPECTRA APPROXIMATION  #####################################                     

E01=0
E02=0

def Young(E, E0, A, R, FWHM, B, K):
    #R=1
    I_el = A*np.exp((-(1-R)*2.77259*(E-E0)/(FWHM+0.001)*2))/(R*(E-E0)**2+((FWHM+0.001)/2)**2)
    I_inel = B*(np.pi-2*np.arctan(2*(E-E0)/(FWHM+0.001)))
    I_tail = np.exp(-K/(np.sqrt(E)+0.001))    
    return I_el+I_inel*I_tail

def twoCompTargetFitting(E, A1, FWHM1, B1, K1, A2, FWHM2, B2, K2):
    return Young(E, E01, A1, 1, FWHM1, B1, K1) + Young(E, E02, A2, 1, FWHM2, B2, K2)


"""
param_bounds = ([0,0,0,0,0,0,0,0],[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])
pars, covariance = curve_fit(twoCompTargetFitting, spectrum_en, spectrum_int, method = 'dogbox', maxfev=500000, bounds=param_bounds, loss='linear', ftol=1E-9)
    
inel_Au = Young(spectrum_en, E02, 0, 1, pars[5], pars[6], pars[7])
El_Au = Young(spectrum_en, E02, pars[4], 1, pars[5], 0, pars[7])
inel_Pd = Young(spectrum_en, E01, 0, 1, pars[1], pars[2], pars[3])
El_Pd = Young(spectrum_en, E01, pars[0], 1, pars[1], 0, pars[3])


fitted = twoCompTargetFitting(spectrum_en, pars[0], pars[1],pars[2],pars[3], pars[4], pars[5], pars[6], pars[7])
    
Au_part = Young(spectrum_en, E02, pars[4], 1, pars[5], pars[6], pars[7])
Au_part = norm (Au_part)

Pd_part_fit = fitted - Au_part

Au_conc_fit = calcConcFor2ElementsSpectrum(peak(Pd_part_fit), 1)

Au_ref_interp = np.interp(spectrum_en,reference_en, reference_int)
Au_ref_interp = norm(Au_ref_interp)

Pd_part_ref = spectrum_int - Au_ref_interp

Au_conc_ref = calcConcFor2ElementsSpectrum(peak(Pd_part_ref),1)  

#Au_conc_ref = (1-peak(Pd_part_ref)*sigma_Au/sigma_Pd/(1+peak(Pd_part_ref)*sigma_Au/sigma_Pd))*100

Au_conc_refByPseudoArea = calcConcFor2ElementsSpectrum(peak(Pd_part_ref)*dE_Pd,1*dE_Au)  

syn_cons = calcConcFor2ElementsSpectrum(sum(El_Pd), sum(El_Au))
    
"""

#####################################  CROSS-SECTION CALCULATION   #####################################

#     /$$$$$$  /$$$$$$$   /$$$$$$   /$$$$$$   /$$$$$$           /$$$$$$  /$$$$$$$$  /$$$$$$  /$$$$$$$$ /$$$$$$  /$$$$$$  /$$   /$$
#    /$$__  $$| $$__  $$ /$$__  $$ /$$__  $$ /$$__  $$         /$$__  $$| $$_____/ /$$__  $$|__  $$__/|_  $$_/ /$$__  $$| $$$ | $$
#   | $$  \__/| $$  \ $$| $$  \ $$| $$  \__/| $$  \__/        | $$  \__/| $$      | $$  \__/   | $$     | $$  | $$  \ $$| $$$$| $$
#   | $$      | $$$$$$$/| $$  | $$|  $$$$$$ |  $$$$$$  /$$$$$$|  $$$$$$ | $$$$$   | $$         | $$     | $$  | $$  | $$| $$ $$ $$
#   | $$      | $$__  $$| $$  | $$ \____  $$ \____  $$|______/ \____  $$| $$__/   | $$         | $$     | $$  | $$  | $$| $$  $$$$
#   | $$    $$| $$  \ $$| $$  | $$ /$$  \ $$ /$$  \ $$         /$$  \ $$| $$      | $$    $$   | $$     | $$  | $$  | $$| $$\  $$$
#   |  $$$$$$/| $$  | $$|  $$$$$$/|  $$$$$$/|  $$$$$$/        |  $$$$$$/| $$$$$$$$|  $$$$$$/   | $$    /$$$$$$|  $$$$$$/| $$ \  $$
#    \______/ |__/  |__/ \______/  \______/  \______/          \______/ |________/ \______/    |__/   |______/ \______/ |__/  \__/
                                                                                                                                                                                                                                                                                                                                                                                 
#####################################  CROSS-SECTION CALCULATION   #####################################

# Constants
epsilon_0 = 8.854187817e-12  # Vacuum permittivity (F/m)
e = 1.602176634e-19  # Elementary charge (C)
a0 = 5.29177210903e-11  # Bohr radius (m)


# Global variables
m = [0, 0]  # Masses of projectile and target
z = [0, 0]  # Atomic numbers of projectile and target
E = 0  # Energy of the projectile
a = 0  # Screening length
Z = 0  # Product of atomic numbers
U = 0  # Potential energy
c = 0  # Curvature parameter
B = 0  # Impact parameter
o = 0  # Scattering angle
e1 = 0  # Reduced energy
R = 0  # Reduced distance
C1, C2, C3, C4, C5 = 0, 0, 0, 0, 0  # Constants for potential
s1, s2, s3, s4 = 0, 0, 0, 0  # Screening function parameters
d1, d2, d3, d4 = 0, 0, 0, 0  # Screening function parameters
pot = 1  # Potential type

def _element(ta, por):
    global m, z
    
    """
    elements = [
        (1, 1), (2, 1), (3, 1), (4, 2), (3, 2), (6.94, 3), (9.01, 4), (10.81, 5),
        (12.011, 6), (14, 7), (16, 8), (19, 9), (20.18, 10), (20, 10), (22.99, 11),
        (24.305, 12), (26.9815, 13), (28.086, 14), (30.974, 15), (32.066, 16),
        (35.4527, 17), (39.948, 18), (39.098, 19), (40.078, 20), (44.956, 21),
        (47.867, 22), (50.941, 23), (51.996, 24), (54.938, 25), (55.845, 26),
        (58.933, 27), (58.693, 28), (63.546, 29), (65.39, 30), (69.723, 31),
        (72.61, 32), (74.922, 33), (78.96, 34), (79.904, 35), (83.8, 36),
        (85.4678, 37), (87.62, 38), (88.906, 39), (91.224, 40), (92.906, 41),
        (95.94, 42), (97, 43), (101.07, 44), (102.906, 45), (106.42, 46),
        (107.868, 47), (112.411, 48), (114.818, 49), (118.71, 50), (121.76, 51),
        (127.6, 52), (126.904, 53), (131.29, 54), (132.905, 55), (137.327, 56),
        (138.906, 57), (140.116, 58), (140.908, 59), (144.24, 60), (145, 61),
        (150.36, 62), (151.964, 63), (157.25, 64), (158.925, 65), (162.5, 66),
        (164.93, 67), (167.26, 68), (168.934, 69), (173.04, 70), (174.967, 71),
        (178.46, 72), (180.948, 73), (183.84, 74), (186.207, 75), (190.23, 76),
        (192.217, 77), (195.078, 78), (196.967, 79), (200.59, 80), (204.383, 81),
        (207.2, 82), (208.98, 83), (210, 84), (210, 85), (222, 86), (223, 87),
        (226, 88), (227, 89), (232, 90), (231, 91), (238, 92)
    ]
    #m[por], z[por] = elements[ta]
    """
    
    z[por]=next((el for el in elements if el.symbol == ta), None).atomic_number #projectile atomic number
    m[por]=next((el for el in elements if el.symbol == ta), None).atomic_weight #projectile atomic mass
    

def _potenc(r0):
    global U, R
    U = Z * 23.0707e-20 * (s1 * math.exp(-d1 * R) + s2 * math.exp(-d2 * R) + s3 * math.exp(-d3 * R) + s4 * math.exp(-d4 * R)) / (R * a)
    return 0

def _pric(r0):
    global B
    _potenc(r0)
    if (1 - U / E) < 0:
        return 1
    else:
        P = r0 * math.pow(1 - U / E, 0.5)
        B = P / a
        return 0

def _criv(r0):
    global c
    c = 2 * (E - U) * r0 / (a * U + Z * 23.0707e-20 * (s1 * d1 * math.exp(-d1 * R) + s2 * d2 * math.exp(-d2 * R) + s3 * d3 * math.exp(-d3 * R) + s4 * d4 * math.exp(-d4 * R)))
    return 0

def _res():
    global e1, B, R, o
    be = (C2 + math.pow(e1, 0.5)) / (C3 + math.pow(e1, 0.5))
    A0 = 2 * (1 + C1 * math.pow(e1, -0.5)) * e1 * math.pow(B, be)
    G = (C5 + e1) / ((C4 + e1) * (math.pow(1 + A0 * A0, 0.5) - A0))
    d = (B + c + A0 * (R - B) / (1 + G)) / (R + c) - math.cos(o / 2)
    return d

def _approach( E0):
    global E, a, Z, e1, R, q
    E = 1.6021766e-12 * E0 / (1 + m[0] / m[1])
    z0 = math.pow(z[0], 0.5) + math.pow(z[1], 0.5)
    a = 0.8853 * 0.529e-8 / math.pow(z0, 0.666666666)
    if pot == 1 or pot == 2:
        z0 = math.pow(z[0], 0.23) + math.pow(z[1], 0.23)
        a = 0.88534 * 0.529e-8 / z0
    e1 = a * E / (z[0] * z[1] * 23.0707e-20)
    Z = z[0] * z[1]
    x1 = 0
    x2 = 5e-8
    for i in range(1, 41):
        y = (x1 + x2) / 2
        R = y / a
        q = _pric(y)
        if q == 0:
            q = _criv(y)
            re = _res()
            if re > 0:
                x2 = y
            else:
                x1 = y
        if q == 1:
            x1 = y
    y = (x1 + x2) / 2
    return y

def __vybit( E0, o2, od):
    global o, En1, dif1
    hi = math.pi - 2 * o2
    o1 = math.atan(m[1] * math.sin(hi) / (m[0] + m[1] * math.cos(hi)))
    if o1 < 0:
        o1 = math.pi + o1
    #print(f"Scattering angle: {o1 * 180 / math.pi:.4f} degrees")
    if (m[1] / m[0]) < 1:
        if o2 > ((math.pi / 4) - (math.asin(m[1] / m[0])) / 2):
            E1 = E0 * math.pow((math.cos(o1) + math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1), 2), 0.5)) / (1 + m[1] / m[0]), 2)
        else:
            E1 = E0 * math.pow((math.cos(o1) - math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1), 2), 0.5)) / (1 + m[1] / m[0]), 2)
    else:
        E1 = E0 * math.pow((math.cos(o1) + math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1), 2), 0.5)) / (1 + m[1] / m[0]), 2)
    #print(f"Energy loss: {E0 - E1:.0f} eV")
    En1 = E0 - E1
    o = math.atan(math.sin(o1) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(o1) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
    if o < 0:
        o = math.pi + o
    #print(f"Scattering angle: {o * 180 / math.pi:.2f} degrees")
    r0 = _approach(E0)
    #print(f"__Approach distance: {r0 * 1e8:.5f} Å")
    q = _pric(r0)
    #print(f"Impact parameter: {B * a * 1e8:.5f} Å")
    orm = o2 - od / 2
    orp = o2 + od / 2
    if o2 + od / 2 > math.pi / 2 or orm <= 0:
        print("No solution")
        dif1 = -1
    else:
        hi = math.pi - 2 * orm
        o1 = math.atan(m[1] * math.sin(hi) / (m[0] + m[1] * math.cos(hi)))
        if o1 < 0:
            o1 = math.pi + o1
        if (m[1] / m[0]) < 1:
            if orm > (math.pi / 4 - math.asin(m[1] / m[0]) / 2):
                E1 = E0 * math.pow((math.cos(o1) + math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1), 2), 0.5)) / (1 + m[1] / m[0]), 2)
            else:
                E1 = E0 * math.pow((math.cos(o1) - math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1), 2), 0.5)) / (1 + m[1] / m[0]), 2)
        else:
            E1 = E0 * math.pow((math.cos(o1) + math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1), 2), 0.5)) / (1 + m[1] / m[0]), 2)
        o = math.atan(math.sin(o1) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(o1) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
        if o < 0:
            o = math.pi + o
        r0 = _approach(E0)
        q = _pric(r0)
        p1 = B * a * 1e8
        hi = math.pi - 2 * orp
        o1 = math.atan(m[1] * math.sin(hi) / (m[0] + m[1] * math.cos(hi)))
        if o1 < 0:
            o1 = math.pi + o1
        if (m[1] / m[0]) < 1:
            if orp > (math.pi / 4 - math.asin(m[1] / m[0]) / 2):
                E1 = E0 * math.pow((math.cos(o1) + math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1), 2), 0.5)) / (1 + m[1] / m[0]), 2)
            else:
                E1 = E0 * math.pow((math.cos(o1) - math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1), 2), 0.5)) / (1 + m[1] / m[0]), 2)
        else:
            E1 = E0 * math.pow((math.cos(o1) + math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1), 2), 0.5)) / (1 + m[1] / m[0]), 2)
        o = math.atan(math.sin(o1) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(o1) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
        if o < 0:
            o = math.pi + o
        r0 = _approach(E0)
        q = _pric(r0)
        p2 = B * a * 1e8
       # print(f"Difference: {abs(p1**2 - p2**2):.7f}")
        dif1 = abs(p1**2 - p2**2)

def get_cross_section(incident_symbol, E0, o1, od, target_symbol):
    
    """
    returns cross section of the scattering process for given incident and target elements
    incident_symbol - symbol of the incident element
    E0 - energy of the incident particle
    o1 - scattering angle
    od - spread of the scattering angle
    target_symbol - symbol of the target element  
    """
    
    o1 = o1 * math.pi / 180
    od = od * math.pi / 180
    
    global o, En1, En2, dif1, dif2
    global C1, C2, C3, C4, C5, s1, s2, s3, s4, d1, d2, d3, d4

    #print (incident_symbol+" -> "+target_symbol)
    _element(incident_symbol, 0)  # Example: Neon as projectile
    _element(target_symbol, 1)  # Example: Tungsten as target
    
    if pot == 0:
        s1, s2, s3, s4 = 0.35, 0.55, 0.1, 0
        d1, d2, d3, d4 = 0.3, 1.2, 6, 0
        C1, C2, C3, C4, C5 = 0.6743, 0.009611, 0.005175, 6.314, 10
    elif pot == 1:
        s1, s2, s3, s4 = 0.028171, 0.28022, 0.50986, 0.18175
        d1, d2, d3, d4 = 0.20162, 0.40290, 0.94229, 3.1998
        C1, C2, C3, C4, C5 = 0.99229, 0.011615, 0.0071222, 9.3066, 14.813
    elif pot == 2:
        s1, s2, s3, s4 = 0.190945, 0.473674, 0.335381, 0
        d1, d2, d3, d4 = 0.278544, 0.637174, 1.919249, 0
        C1, C2, C3, C4, C5 = 1.0144, 0.235809, 0.126, 6.9350, 8.3550
    

    E1 = E0 * math.pow((math.cos(o1) + math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1), 2), 0.5)) / (1 + m[1] / m[0]), 2)
    #print(f"Energy after scattering: {E1:.0f} eV")
    En1 = E1
    E2 = E0 * math.pow((math.cos(o1 / 2) + math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1 / 2), 2), 0.5)) / (1 + m[1] / m[0]), 4)
    #print(f"Double scattering energy: {E2:.0f} eV")
    o = math.atan(math.sin(o1) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(o1) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
    if o < 0:
        o = math.pi + o
    #print(f"Scattering angle: {o * 180 / math.pi:.2f} degrees")
    r0 = _approach(E0)
    #print(f"__Approach distance: {r0 * 1e8:.5f} Å")
    q = _pric(r0)
    pc1 = B * a * 1e8
    #print(f"Impact parameter: {pc1:.5f} Å")
    if o1 - od / 2 <= 0 or o1 + od / 2 >= 180 or ((math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1 + od / 2), 2))) < 0:
        print("No solution")
        dif1 = -1
    else:
        orm = o1 - od / 2
        orp = o1 + od / 2
        E1 = E0 * math.pow((math.cos(orm) + math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(orm), 2), 0.5)) / (1 + m[1] / m[0]), 2)
        o = math.atan(math.sin(orm) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(orm) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
        if o < 0:
            o = math.pi + o
        r0 = _approach(E0)
        q = _pric(r0)
        p1 = B * a * 1e8
        E1 = E0 * math.pow((math.cos(orp) + math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(orp), 2), 0.5)) / (1 + m[1] / m[0]), 2)
        o = math.atan(math.sin(orp) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(orp) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
        if o < 0:
            o = math.pi + o
        r0 = _approach(E0)
        q = _pric(r0)
        p2 = B * a * 1e8
        #print(f"Difference: {abs(p1**2 - p2**2):.7f}")
        dif1 = abs(p1**2 - p2**2)

    if (m[1] / m[0]) < 1:
        E1 = E0 * math.pow((math.cos(o1) - math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1), 2), 0.5)) / (1 + m[1] / m[0]), 2)
        print(f"Complementary energy: {E1:.0f} eV")
        En2 = E1
        E2 = E0 * math.pow((math.cos(o1 / 2) - math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1 / 2), 2), 0.5)) / (1 + m[1] / m[0]), 4)
        print(f"Complementary double scattering energy: {E2:.0f} eV")
        o = math.atan(math.sin(o1) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(o1) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
        if o < 0:
            o = math.pi + o
        print(f"Complementary scattering angle: {o * 180 / math.pi:.2f} degrees")
        r0 = _approach(E0)
        print(f"Complementary __approach distance: {r0 * 1e8:.5f} Å")
        q = _pric(r0)
        pc2 = B * a * 1e8
        print(f"Complementary impact parameter: {pc2:.5f} Å")

        if o1 - od / 2 <= 0 or o1 + od / 2 >= 180 or ((math.pow(m[1] / m[0], 2) - math.pow(math.sin(o1 + od / 2), 2))) < 0:
            print("No solution")
            dif2 = -1
        else:
            orm = o1 - od / 2
            orp = o1 + od / 2
            if orp >= 180:
                orp = 360 - orp
            E1 = E0 * math.pow((math.cos(orm) - math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(orm), 2), 0.5)) / (1 + m[1] / m[0]), 2)
            o = math.atan(math.sin(orm) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(orm) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
            if o < 0:
                o = math.pi + o
            r0 = _approach(E0)
            q = _pric(r0)
            p1 = B * a * 1e8
            E1 = E0 * math.pow((math.cos(orp) - math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(orp), 2), 0.5)) / (1 + m[1] / m[0]), 2)
            o = math.atan(math.sin(orp) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(orp) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
            if o < 0:
                o = math.pi + o
            r0 = _approach(E0)
            q = _pric(r0)
            p2 = B * a * 1e8
            print(f"Complementary difference: {abs(p1**2 - p2**2):.7f}")
            dif2 = abs(p1**2 - p2**2)

            orm = o1 - 0.00001
            orp = o1 + 0.00001
            E1 = E0 * math.pow((math.cos(orm) - math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(orm), 2), 0.5)) / (1 + m[1] / m[0]), 2)
            o = math.atan(math.sin(orm) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(orm) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
            if o < 0:
                o = math.pi + o
            r0 = _approach(E0)
            q = _pric(r0)
            p1 = B * a * 1e8
            E1 = E0 * math.pow((math.cos(orp) - math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(orp), 2), 0.5)) / (1 + m[1] / m[0]), 2)
            o = math.atan(math.sin(orp) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(orp) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
            if o < 0:
                o = math.pi + o
            r0 = _approach(E0)
            q = _pric(r0)
            p2 = B * a * 1e8
            print(f"Complementary differential cross-section: {abs((p2 - p1) * (p1 + p2) / (2 * math.sin(o1) * 2e-5)):.7f}")
    else:
        En2 = -1

    orm = o1 - 0.00001
    orp = o1 + 0.00001
    E1 = E0 * math.pow((math.cos(orm) + math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(orm), 2), 0.5)) / (1 + m[1] / m[0]), 2)
    o = math.atan(math.sin(orm) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(orm) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
    if o < 0:
        o = math.pi + o
    r0 = _approach(E0)
    q = _pric(r0)
    p1 = B * a * 1e8
    E1 = E0 * math.pow((math.cos(orp) + math.pow(math.pow(m[1] / m[0], 2) - math.pow(math.sin(orp), 2), 0.5)) / (1 + m[1] / m[0]), 2)
    o = math.atan(math.sin(orp) * math.pow(2 * m[0] * E1, 0.5) / ((m[0] * (math.cos(orp) * math.pow(2 * m[0] * E1, 0.5) / m[0] - math.pow(2 * m[0] * E0, 0.5) / (m[0] + m[1])))))
    if o < 0:
        o = math.pi + o
    r0 = _approach(E0)
    q = _pric(r0)
    p2 = B * a * 1e8
    #print(f"Differential cross-section: {abs((p1 - p2) * (p1 + p2) / (2 * math.sin(o1) * 2e-5)):.7f}")
    #print ("--------------------------------")
    return abs((p1 - p2) * (p1 + p2) / (2 * math.sin(o1) * 2e-5))



