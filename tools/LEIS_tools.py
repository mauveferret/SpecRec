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
from scipy.optimize import curve_fit
import math, re, os
#from mendeleev import get_all_elements

######################    GLOBAL SETTINGS   ############################


step = 2.0 #energy step, eV
Emin = 300.0 #minimal energy, eV
Emax = 30000.0 #maximal energy, eV

######################    GLOBAL SETTINGS   ############################

total_num_elem = 93
def get_element_info_by_atomic_number(atomic_number: int):
    """
    Returns the mass and symbol (int, str) of the element given its atomic number.
    """
    
    # Define a dictionary with atomic numbers as keys and tuples of (mass, symbol) as values
    elements_data = {
        1: (1.008, 'H'), 2: (4.0026, 'He'), 3: (6.94, 'Li'), 4: (9.0122, 'Be'), 5: (10.81, 'B'),
        6: (12.011, 'C'), 7: (14.007, 'N'), 8: (15.999, 'O'), 9: (18.998, 'F'), 10: (20.180, 'Ne'),
        11: (22.989, 'Na'), 12: (24.305, 'Mg'), 13: (26.982, 'Al'), 14: (28.085, 'Si'), 15: (30.974, 'P'),
        16: (32.06, 'S'), 17: (35.45, 'Cl'), 18: (39.948, 'Ar'), 19: (39.098, 'K'), 20: (40.078, 'Ca'),
        21: (44.956, 'Sc'), 22: (47.867, 'Ti'), 23: (50.942, 'V'), 24: (51.996, 'Cr'), 25: (54.938, 'Mn'),
        26: (55.845, 'Fe'), 27: (58.933, 'Co'), 28: (58.693, 'Ni'), 29: (63.546, 'Cu'), 30: (65.38, 'Zn'),
        31: (69.723, 'Ga'), 32: (72.63, 'Ge'), 33: (74.922, 'As'), 34: (78.971, 'Se'), 35: (79.904, 'Br'),
        36: (83.798, 'Kr'), 37: (85.468, 'Rb'), 38: (87.62, 'Sr'), 39: (88.906, 'Y'), 40: (91.224, 'Zr'),
        41: (92.906, 'Nb'), 42: (95.95, 'Mo'), 43: (98, 'Tc'), 44: (101.07, 'Ru'), 45: (102.91, 'Rh'),
        46: (106.42, 'Pd'), 47: (107.87, 'Ag'), 48: (112.41, 'Cd'), 49: (114.82, 'In'), 50: (118.71, 'Sn'),
        51: (121.76, 'Sb'), 52: (127.6, 'Te'), 53: (126.9, 'I'), 54: (131.29, 'Xe'), 55: (132.91, 'Cs'),
        56: (137.33, 'Ba'), 57: (138.91, 'La'), 58: (140.12, 'Ce'), 59: (140.91, 'Pr'), 60: (144.24, 'Nd'),
        61: (145, 'Pm'), 62: (150.36, 'Sm'), 63: (151.96, 'Eu'), 64: (157.25, 'Gd'), 65: (158.93, 'Tb'),
        66: (162.5, 'Dy'), 67: (164.93, 'Ho'), 68: (167.26, 'Er'), 69: (168.93, 'Tm'), 70: (173.04, 'Yb'),
        71: (174.97, 'Lu'), 72: (178.49, 'Hf'), 73: (180.95, 'Ta'), 74: (183.84, 'W'), 75: (186.21, 'Re'),
        76: (190.23, 'Os'), 77: (192.22, 'Ir'), 78: (195.08, 'Pt'), 79: (196.97, 'Au'), 80: (200.59, 'Hg'),
        81: (204.38, 'Tl'), 82: (207.2, 'Pb'), 83: (208.98, 'Bi'), 84: (209, 'Po'), 85: (210, 'At'),
        86: (222, 'Rn'), 87: (223, 'Fr'), 88: (226, 'Ra'), 89: (227, 'Ac'), 90: (232.04, 'Th'),
        91: (231.04, 'Pa'), 92: (238.03, 'U')
    }
    global total_num_elem 
    total_num_elem = len(elements_data)

    return elements_data[atomic_number]

#elements = get_all_elements()
#print("periodic chemical elements database is LOADED")

class spectrum:
    
    def __init__(self, spectrum_path: str, filter_window_length: int=-1, step: float = 2.0):
        """
        Constructor of the spectrum class.
        spectrum_path - path to the file with the spectrum
        filter_window_length - length of the filter window for smoothing of the spectrum in eV
        step - energy step in eV
        """
        self.__step = step
        self.import_spectrum(spectrum_path, filter_window_length)
        #self.do_elemental_analysis()
    
    @property
    def spectrum_en(self):
        return self.__spectrum_en

    @property
    def spectrum_int(self):
        return self.__spectrum_int
    
    @property
    def calc_name(self):
        return self.__calc_name
    
    @calc_name.setter
    def calc_name(self, value: str):
        self.__calc_name = value
           
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
        return self.__M0
    
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
    
    @property
    def peaks(self):
        """
        Peaks of the spectrum
        """
        return self.__peaks
    
    @property
    def target_components(self):
        """
        Chemical elements of the peaks
        """
        return self.__target_components
    
    @property
    def cross_sections(self):
        """
        Cross sections of the peaks
        """
        return self.__cross_sections
    
    @property
    def dBetas(self):    
        """
        Delta of scattering angle in degrees for specific bin size dE of the analyzer at energy position
        corresponding to mu==M_target/M_incident
        """
        return self.__dBetas
    
    @property
    def elem_conc_by_corrI(self):
        """
        Relative surface concentration of the component on the LEIS spectrum in %
        """
        return self.__elem_conc_by_corrI
    
    @property
    def elem_conc_by_I(self):
        """
        Relative surface concentration of the component on the LEIS spectrum in %
        """
        return self.__elem_conc_by_I
    
    @property
    def elem_conc_by_S(self):
        """
        Relative surface concentration of the component on the LEIS spectrum in %
        """
        return self.__elem_conc_by_S

    def import_spectrum(self, spectrum_path: str, filter_window_length=-1):  
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
        
        #print(self.__calc_name)
        
        # find letter strings and save its indexes
        __indexes_letter_strings = []
        for i in range(0, len(lines)):
            if  any(c.isalpha() and not ("e" in c) for c in lines[i]) or ("*" in lines[i]) or lines[i]=='':
                __indexes_letter_strings.append(lines[i])
        if "sim" in self.__calc_name:
            # get initial params from the filename
            self.__incident_atom = re.sub(r'\d', '', spectrum_path.split(os.sep)[-1].split("_")[1].split("keV")[0]).replace(".","")
            self.__M0 = get_mass_by_element(self.__incident_atom)
            
            self.__E0 = float(spectrum_path.split(os.sep)[-1].split("_")[1].split("keV")[0].split(self.__incident_atom)[1])*1000
            self.__theta = float(spectrum_path.split(os.sep)[-1].split("_")[1].split("keV")[1].split("deg")[0])
            try:
                self.__dTheta = float(spectrum_path.split(os.sep)[-1].split("_")[1].split("keV")[1].split("deg")[1])
            except:
                print("ERROR during spectrum import. Can't find dTheta in the filename")
                self.__dTheta = 1.0
        elif ("exp" in self.__calc_name or "exp" in spectrum_path) and not "exp_ref" in self.__calc_name:
            # parameters are valid ONLY for data obtained on
            # the Large-mass monocromator "Mephi" aka "Crocodile" LEIS facility  
            self.__incident_atom = spectrum_path.split(os.sep)[-1].split("+")[0].split("-")[-1]
            self.__M0 = get_mass_by_element(self.__incident_atom)
            self.__E0 = float(spectrum_path.split(os.sep)[-1].split("+")[1].split("keV")[0].strip())*1000
            self.__theta = 32 # scattering angle is usually fixed at 32 degrees
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
            if "sim" in self.__calc_name:
                raw_spectrum_en[i] = float(data[0])
            elif "exp" in self.__calc_name or "exp" in spectrum_path:
                raw_spectrum_en[i] = float(data[0])*1000
            raw_spectrum_int[i] = float(data[1])

        # do interpolation with new energy step and normalization to 1 in range (Emin, Emax)
        self.__spectrum_en = np.arange(0, raw_spectrum_en[-1], self.__step)
        # scaling range in eV (influence spectra normalization in Web charts and output files)
        self.__spectrum_int = np.interp(self.__spectrum_en,raw_spectrum_en, raw_spectrum_int)
        global Emax
        Emax = self.__spectrum_en[-1]-100
        self.__spectrum_int /= max(self.__spectrum_int[int(Emin/step):int(self.__spectrum_en[-1]-100/step)])
        if filter_window_length > 0:
            self._smooth(filter_window_length)
   
    def _smooth(self, filter_window_length: int):
            self.__spectrum_int = scipy.signal.savgol_filter(self.__spectrum_int, int(filter_window_length/self.__step), 3)

    def get_target_mass_by_energy(self, E1: float):
        """
        Returns mass of the target element by energy of the scattered particle E1
        """
        return get_target_mass_by_energy(self.__theta, self.__M0, self.__E0, E1)
    
    def get_element_by_mass(self, mass: float):
        """
        Returns Symbol of chemical element with the closest atomic weight to the given mass
        """
        return get_element_by_mass(mass)
    
    def get_dBeta(self, mu: float, dE: float):
        """
        Returns delta of scattering angle in degrees for specific bin size dE of the analyzer at energy position
        corresponding to mu==M_target/M_incident
        """
        return get_dBeta(self.__E0, self.__theta, mu, dE)
    
    def get_dE(self, mu: float):
        """
        Returns delta of energy in eV for specific detection angle of the analyzer at energy position
        corresponding to mu==M_target/M_incident
        """
        return get_dE(self.__E0, self.__theta, mu, self.dTheta)
    
    def get_cross_section(self, target_symbol: str):
        """
        Returns cross section of the target element by its symbol
        """
        return get_cross_section(self.__incident_atom, self.__E0, self.__theta, self.__dTheta, target_symbol)
    
    def do_elemental_analysis(self):
        """
        Method to find peaks and corresponding elements in the spectrum
        """
        peaks, _ = scipy.signal.find_peaks(self.__spectrum_int, prominence=0.04, width=5, distance=200)
        self.__peaks = peaks
        target_masses = [self.get_target_mass_by_energy(peak) for peak in self.__spectrum_en[peaks]]
        target_components = [self.get_element_by_mass(mass) for mass in target_masses]
        
        self.__dBetas = [ self.get_dBeta(mass/self.__M0, self.__step) for mass in target_masses]
        dEs = [self.get_dE(mass/self.M0) for mass in target_masses]
        self.__cross_sections = [self.get_cross_section(component) for component in target_components]
        
        for i in range (0,len(target_components)):
            if "Ir" in target_components[i] or "At" in target_components[i] or "Re" in target_components[i] or "Pt" in target_components[i]:
                target_components[i]="Au"
                target_masses[i] = get_mass_by_element(target_components[i])
            if "Cd" in target_components[i] or "Ag" in target_components[i]:
                target_components[i]="Pd"
                target_masses[i] = get_mass_by_element(target_components[i])
        
        int_total = 0
        for i in range (0,len(target_components)):          
            int_total += self.__spectrum_int[peaks[i]]/(self.__cross_sections[i]*self.__dBetas[i])
            #int_total += self.__spectrum_int[peaks[i]]*dEs[i]/(self.__cross_sections[i])

        self.__elem_conc_by_corrI = np.zeros(len(target_components))
        for i in range (0,len(target_components)):      
            self.__elem_conc_by_corrI[i] = self.__spectrum_int[peaks[i]]/(self.__cross_sections[i]*self.__dBetas[i])/int_total*100
            #self.__elem_conc_by_corrI[i] = self.__spectrum_int[peaks[i]]*dEs[i]/(self.__cross_sections[i])/int_total*100

        int_total = 0
        for i in range (0,len(target_components)):          
            int_total += self.__spectrum_int[peaks[i]]/(self.__cross_sections[i])
        
        self.__elem_conc_by_I = np.zeros(len(target_components))
        for i in range (0,len(target_components)):      
            self.__elem_conc_by_I[i] = self.__spectrum_int[peaks[i]]/(self.__cross_sections[i])/int_total*100

        area_total = 0
        for i in range (0,len(target_components)):        
            area_total += int(sum(self.__spectrum_int[int(peaks[i]-dEs[i]/2):int(peaks[i]+dEs[i]/2)]))/self.__cross_sections[i]
        
        self.__elem_conc_by_S = np.zeros(len(target_components))
        for i in range (0,len(target_components)):      
            self.__elem_conc_by_S[i] = sum(self.__spectrum_int[int(peaks[i]-dEs[i]/2):int(peaks[i]+dEs[i]/2)])/(self.__cross_sections[i])/area_total*100         
            
        self.__target_components = target_components
        
        return peaks, target_components

#################################    COMMON  #####################################


#     /$$$$$$   /$$$$$$  /$$      /$$ /$$      /$$  /$$$$$$  /$$   /$$
#    /$$__  $$ /$$__  $$| $$$    /$$$| $$$    /$$$ /$$__  $$| $$$ | $$
#   | $$  \__/| $$  \ $$| $$$$  /$$$$| $$$$  /$$$$| $$  \ $$| $$$$| $$
#   | $$      | $$  | $$| $$ $$/$$ $$| $$ $$/$$ $$| $$  | $$| $$ $$ $$
#   | $$      | $$  | $$| $$  $$$| $$| $$  $$$| $$| $$  | $$| $$  $$$$
#   | $$    $$| $$  | $$| $$\  $ | $$| $$\  $ | $$| $$  | $$| $$\  $$$
#   |  $$$$$$/|  $$$$$$/| $$ \/  | $$| $$ \/  | $$|  $$$$$$/| $$ \  $$
#    \______/  \______/ |__/     |__/|__/     |__/ \______/ |__/  \__/
                                                                  
                                                                  
#################################    COMMON  #####################################

        
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

def get_standart_deviation(data):
    """
    method for calculating average and standart deviation of the data
    returns tuple (average, standart_deviation)
    """
    n = len(data)
    average = sum(data)/n
    standart_deviation = 0
    for value in data:
        standart_deviation+=value**2
    standart_deviation=1/n*np.sqrt(standart_deviation-n*average**2)
    return average, standart_deviation
 
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
    #masses = {el: el.atomic_weight for el in elements if el.atomic_weight}
    masses = {get_element_info_by_atomic_number(i)[1]:get_element_info_by_atomic_number(i)[0] for i in range(1, total_num_elem)}

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
    #return next((el for el in elements if el.symbol == element_symbol), None).atomic_weight 
    
    return next((get_element_info_by_atomic_number(i)[0] for i in range(1, total_num_elem) 
                 if get_element_info_by_atomic_number(i)[1] == element_symbol), None)
    

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

def get_sensitivity_factor(E0:float, incident_element:str, target_element:str, theta:float, dTheta:float):
    """
    Returns sensitivity factor for the given incident and target elements
    """
    mu = get_mass_by_element(target_element)/get_mass_by_element(incident_element)
    return 1/(get_dBeta(E0, theta, mu, get_dE(E0, theta, mu, dTheta))*get_cross_section(incident_element, E0, theta, dTheta, target_element))

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
    """
    Method to plot the map of dBeta values in dependence of scattering angle and mu
    """
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

def plot_spectrum_with_concs(spectrum: spectrum, title = None):
    """
    Method to plot the spectrum with quantified peaks
    """
    plt.plot(spectrum.spectrum_en[int(Emin/spectrum.step):]/1000, spectrum.spectrum_int[int(Emin/spectrum.step):], '-', linewidth=2, label=spectrum.calc_name) 
    plt.plot(spectrum.spectrum_en[spectrum.peaks]/1000, spectrum.spectrum_int[spectrum.peaks], "o", color='red')

    i=0
    for x,y in zip(spectrum.spectrum_en[spectrum.peaks]/1000,spectrum.spectrum_int[spectrum.peaks]):


        label = str(spectrum.target_components[i])+"\n"+str(spectrum.elem_conc_by_corrI[i])[0:4]+"% "+str(spectrum.elem_conc_by_S[i])[0:4]+"%"
        i+=1
        plt.annotate(label, # this is the text
                    (x,y), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(5,5), # distance from text to points (x,y)
                    ha='center',
                    color='black')
    plt.xlabel('energy, keV', fontsize=12)
    plt.xlim (Emin/1000, Emax/1000)
    plt.ylabel('intensity, a.u.', fontsize=12)
    plt.ylim(0, 1.1)
    plt.minorticks_on
    if title:
        plt.title(title, y=1.01, fontsize=10)
    #plt.legend()
    plt.show()
    
def plot_spectrum(spectrum: spectrum, title = None):
    """
    Method to plot the spectrum with quantified peaks
    """
    plt.plot(spectrum.spectrum_en[int(Emin/spectrum.step):]/1000, spectrum.spectrum_int[int(Emin/spectrum.step):], '-', linewidth=2, label=spectrum.calc_name) 

    plt.xlabel('energy, keV', fontsize=12)
    plt.xlim (Emin/1000, Emax/1000)
    plt.ylabel('intensity, a.u.', fontsize=12)
    plt.ylim(0, 1.1)
    plt.minorticks_on
    if title:
        plt.title(title, y=1.01, fontsize=10)
    plt.legend()
    plt.show()

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


class fitted_spectrum:
    
    __param_bounds = ([0,0,0,0,0,0,0,0],[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])

    
    def __init__(self, spectrum:spectrum, target_element1:str, target_element2:str):
        """
        Constructor of the fitted spectrum class.
        """
        self.__spectrum = spectrum
        self.__target_element1 = target_element1
        self.__target_element2 = target_element2
        self.__E01 = get_energy_by_angle(spectrum.E0,  get_mass_by_element(target_element1)/spectrum.M0, spectrum.theta)
        self.__E02 = get_energy_by_angle(spectrum.E0,  get_mass_by_element(target_element2)/spectrum.M0, spectrum.theta)
        
        pars, covariance = curve_fit(self.__twoCompTargetFitting, 
                                     spectrum.spectrum_en, spectrum.spectrum_int, 
                                     method = 'dogbox', maxfev=500000, bounds=self.__param_bounds, loss='linear', ftol=1E-9)

        self.__pars = pars
        
        
    def __young( E, E0, A, R, FWHM, B, K):
        #R=1
        I_el = A*np.exp((-(1-R)*2.77259*(E-E0)/(FWHM+0.001)*2))/(R*(E-E0)**2+((FWHM+0.001)/2)**2)
        I_inel = B*(np.pi-2*np.arctan(2*(E-E0)/(FWHM+0.001)))
        I_tail = np.exp(-K/(np.sqrt(E)+0.001))    
        return I_el+I_inel*I_tail
    
    def __twoCompTargetFitting(self, E, A1, FWHM1, B1, K1, A2, FWHM2, B2, K2):
        
        def young2( E, E0, A, R, FWHM, B, K):
            #R=1
            I_el = A*np.exp((-(1-R)*2.77259*(E-E0)/(FWHM+0.001)*2))/(R*(E-E0)**2+((FWHM+0.001)/2)**2)
            I_inel = B*(np.pi-2*np.arctan(2*(E-E0)/(FWHM+0.001)))
            I_tail = np.exp(-K/(np.sqrt(E)+0.001))    
            return I_el+I_inel*I_tail
        
        E01 = 14100
        E02 = 14500
        
        # We specify R=1 to use Lorenzian distribution for elastic part of the spectrum
        return young2(E, E01, A1, 1, FWHM1, B1, K1) + young2(E, E02, A2, 1, FWHM2, B2, K2)


    def get_fitted_spectrum(self):
        """
        Method to get the fitted spectrum
        """
        return self.__twoCompTargetFitting(self.__spectrum.spectrum_en, self.__pars[0], self.__pars[1], self.__pars[2], self.__pars[3], self.__pars[4], self.__pars[5], self.__pars[6], self.__pars[7])

    def get_elastic_part(self, element: str):
        """
        Method to get the elastic part of the fitted spectrum
        """
        if element in self.__target_element1:
            return self.__young(self.__spectrum.spectrum_en, self.__E01, self.__pars[0], 1, self.__pars[1], 0, self.__pars[3]) 
        elif element in self.__target_element2: 
            return self.__young(self.__spectrum.spectrum_en, self.__E02, self.__pars[4], 1, self.__pars[5], 0, self.__pars[7])
        else:
            print("ERROR in get_elastic_part: element not found")
            return None
    
    def get_inelastic_part(self, element: str):
        """
        Method to get the inelastic part of the fitted spectrum
        """
        if element in self.__target_element1:
            return self.__young(self.__spectrum.spectrum_en, self.__E01, 0, 1, self.__pars[1], self.__pars[2], self.__pars[3])
        elif element in self.__target_element2:
            return self.__young(self.__spectrum.spectrum_en, self.__E02, 0, 1, self.__pars[5], self.__pars[6], self.__pars[7])
        else:
            print("ERROR in get_inelastic_part: element not found")
            return None


    def get_concentration(self):
        """
        Method to get the concentration of the more heavy element in the fitted spectrum
        """
        int1 = sum(self.get_elastic_part(self.__target_element1))/get_cross_section(self.__spectrum.__incident_atom, 
                                                                                    self.__spectrum.__E0, 
                                                                                    self.__spectrum.__theta, 
                                                                                    self.__spectrum.__dTheta, 
                                                                                    self.__target_element1)
        int2 = sum(self.get_elastic_part(self.__target_element2))/get_cross_section(self.__spectrum.__incident_atom, 
                                                                                    self.__spectrum.__E0, 
                                                                                    self.__spectrum.__theta, 
                                                                                    self.__spectrum.__dTheta, 
                                                                                    self.__target_element2)

        return int2/(int1+int2)*100
        

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
    #z[por]=next((el for el in get_element_info_by_atomic_number if el[1] == ta), None).atomic_number #projectile atomic number
    
    z[por] = next((z for z in range (1, total_num_elem) 
                   if get_element_info_by_atomic_number(z)[1] == ta), None) #projectile atomic number
        
    m[por]=next((get_element_info_by_atomic_number(z)[0] for z in range (1, total_num_elem) 
                 if get_element_info_by_atomic_number(z)[1] == ta), None) #projectile atomic mass

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
