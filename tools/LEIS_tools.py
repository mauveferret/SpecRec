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
FOR _A PARTICULAR PURPOSE. See the GNU General Public License for more details.

If you have questions regarding this program, please contact NEEfimov@mephi.ru
"""

import numpy as np
import matplotlib.pyplot  as plt    
import scipy.signal
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
import math, re, os
#from mendeleev import get_all_elements

######################    GLOBAL SETTINGS   ############################


step = 2.0 #energy step, eV
Emin = 300.0 #minimal energy, eV
Emax = 30000.0 #maximal energy, eV
language = "eng" #language of the output

######################    GLOBAL SETTINGS   ############################

total_num_elem = 93
def get_element_info_by_atomic_number(atomic_number: int):
    """
    Returns the mass and symbol (int, str) of the element given its atomic number.
    """
    
    # Define _a dictionary with atomic numbers as keys and tuples of (mass, symbol) as values
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
    def spectrum_path(self):
        return self.__spectrum_path
    
    @property
    def spectrum_en(self):
        return self.__spectrum_en

    @property
    def spectrum_int(self):
        return self.__spectrum_int
    
    @property
    def spectrum_max(self):
        return self.__spectrum_max
    
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
        return self._M0
    
    @property
    def scattering_angle(self):
        """
        Scattering angle in degrees
        """
        return self.__scattering_angle
    
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
    def elem_conc_by_Icorr(self):
        """
        Relative surface concentration of the component on the LEIS spectrum in %
        """
        return self.__elem_conc_by_Icorr
    
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
        Method to import energy spectrum from _a file located at spectrum_path
        filter windows length is used for smoothing of the spectrum. 
        It has _a default value of -1, which means no smoothing. The positive
        value of filter_window_length will apply Savitzky-Golay filter with 
        polynoms of the 3d order. Filter window length has dimension of energy step [eV].
        """
        self.__spectrum_path = spectrum_path
        try:
            spectrum_file = open(spectrum_path).read()
        except:
            print("ERROR during spectrum import. File not found: "+spectrum_path)
            
        spectrum_file = spectrum_file.replace('\t'," ").replace(",", ".").replace("E","e")
        self.__calc_name = spectrum_path.split(os.sep)[-1].split(".dat")[0]

        lines = spectrum_file.splitlines()
        
        is_not_valid_name = False
        #print(self.__calc_name)
        
        # find letter strings and save its indexes
        __indexes_letter_strings = []
        for i in range(0, len(lines)):
            if  any(c.isalpha() and not ("e" in c) for c in lines[i]) or ("*" in lines[i]) or lines[i]=='':
                __indexes_letter_strings.append(lines[i])
        
        try:
            if "sim" in self.__calc_name or "exp" in self.__calc_name and not "+" in self.__calc_name:
                # get initial params from the filename
                self.__incident_atom = re.sub(r'\d', '', spectrum_path.split(os.sep)[-1].split("_")[1].split("keV")[0]).replace(".","")
                self._M0 = get_mass_by_element(self.__incident_atom)
                
                self.__E0 = float(spectrum_path.split(os.sep)[-1].split("_")[1].split("keV")[0].split(self.__incident_atom)[1])*1000
                self.__scattering_angle = float(spectrum_path.split(os.sep)[-1].split("_")[1].split("keV")[1].split("deg")[0])
                try:
                    self.__dTheta = float(spectrum_path.split(os.sep)[-1].split("_")[1].split("keV")[1].split("deg")[1])
                except:
                    print("ERROR during spectrum import. Can't find dTheta in the filename")
                    self.__dTheta = 1.0
            elif ("exp" in self.__calc_name or "exp" in spectrum_path) and not "exp_ref" in self.__calc_name and not "deg" in self.__calc_name:
                # parameters are valid ONLY for data obtained on
                # the Large-mass monocromator "Mephi" aka "Crocodile" LEIS facility  
                self.__incident_atom = spectrum_path.split(os.sep)[-1].split("+")[0].split("-")[-1]
                self._M0 = get_mass_by_element(self.__incident_atom)
                self.__E0 = float(spectrum_path.split(os.sep)[-1].split("+")[1].split("keV")[0].strip())*1000
                self.__scattering_angle = 32 # scattering angle is usually fixed at 32 degrees
                self.__dTheta = 1.0	
            elif "ex" in spectrum_path and "sim" in spectrum_path:
                # get initial params from the filename
                self.__incident_atom = re.sub(r'\d', '', spectrum_path.split(os.sep)[-2].split("_")[2].split("keV")[0]).replace(".","")
                self._M0 = get_mass_by_element(self.__incident_atom)
                self.__E0 = float(spectrum_path.split(os.sep)[-2].split("_")[2].split("keV")[0].split(self.__incident_atom)[1])*1000
                self.__scattering_angle = float(spectrum_path.split(os.sep)[-2].split("_")[2].split("keV")[1].split("deg")[0])
                self.__dTheta = 1.0
            else:
                 raise Exception("not valid filename")
        except:
            if not "ref" in spectrum_path:
                print("WARNING: spectrum name is not valid. Some scattering parameters can not be loaded. Trying  to guess it for "+spectrum_path)
            is_not_valid_name = True
            self.__incident_atom = "Ne"
            self._M0 = get_mass_by_element(self.__incident_atom)
            self.__E0 = 15000
            self.__scattering_angle = 32 # scattering angle is usually fixed at 32 degrees
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
            if "sim" in self.__calc_name or "sim" in spectrum_path:
                raw_spectrum_en[i] = float(data[0])
            elif "exp" in self.__calc_name or "exp" in spectrum_path or "Спектры" in spectrum_path or True:
                raw_spectrum_en[i] = float(data[0])*1000
            raw_spectrum_int[i] = float(data[1])
        if is_not_valid_name:
            self.__E0 = max (raw_spectrum_en)
        # do interpolation with new energy step and normalization to 1 in range (Emin, Emax)
        self.__spectrum_en = np.arange(0, raw_spectrum_en[-1], self.__step)
        # scaling range in eV (influence spectra normalization in Web charts and output files)
        self.__spectrum_int = np.interp(self.__spectrum_en,raw_spectrum_en, raw_spectrum_int)
        try:
            global Emax
            Emax = self.__spectrum_en[-1]-100
            self.__spectrum_max = max(self.__spectrum_int[int(Emin/step):int((self.__spectrum_en[-1]-100)/step)])
            self.__spectrum_int /= max(self.__spectrum_int[int(Emin/step):int((self.__spectrum_en[-1]-100)/step)])
        except:
            print("ERROR during spectrum import. File not found or corrupted or Emin/Emax are beyond limits: "+spectrum_path)
        if filter_window_length > 0:
            self._smooth(filter_window_length)
   
    def _smooth(self, filter_window_length: int):
            self.__spectrum_int = scipy.signal.savgol_filter(self.__spectrum_int, int(filter_window_length/self.__step), 3)

    def get_target_mass_by_energy(self, E1: float):
        """
        Returns mass of the target element by energy of the scattered particle E1
        """
        return get_target_mass_by_energy(self.__scattering_angle, self._M0, self.__E0, E1)
    
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
        return get_dBeta(self.__E0, self.__scattering_angle, mu, dE)
    
    def get_dE(self, mu: float):
        """
        Returns delta of energy in eV for specific detection angle of the analyzer at energy position
        corresponding to mu==M_target/M_incident
        """
        return get_dE(self.__E0, self.__scattering_angle, mu, self.dTheta)
    
    def get_cross_section(self, target_symbol: str):
        """
        Returns cross section in steradians for scattering on specific target element specified by symbol
        """
        return get_cross_section(self.__incident_atom, self.__E0, self.__scattering_angle, target_symbol)
    
    def do_elemental_analysis(self):
        """
        Method to find peaks and corresponding elements in the spectrum
        """
        peaks, _ = scipy.signal.find_peaks(self.__spectrum_int, prominence=0.05, width=5, distance=60)
        self.__peaks = peaks
        target_masses = [self.get_target_mass_by_energy(peak) for peak in self.__spectrum_en[peaks]]
        target_components = [self.get_element_by_mass(mass) for mass in target_masses]
        
        self.__dBetas = [ self.get_dBeta(mass/self._M0, self.__step) for mass in target_masses]
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

        self.__elem_conc_by_Icorr = np.zeros(len(target_components))
        for i in range (0,len(target_components)):      
            self.__elem_conc_by_Icorr[i] = self.__spectrum_int[peaks[i]]/(self.__cross_sections[i]*self.__dBetas[i])/int_total*100
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

def peak_pos(signal: str):
    """
    """
    min = 0
    i_min = 0
    for i in range(0,len(signal)):
        if signal[i]>min:
            min = signal[i]
            i_min = i
            
    return i_min
        

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
    

def get_dSigma(E0:float, theta:float, mu:float, dE:float):
    """
    method return dOmega of scatted particles in steradians for 
    specific bin size dE of the analyzer at energy position
    cor__responding to mu==M_target/M_incident
    """
    theta10 = get_angle_by_energy(E0, mu, get_energy_by_angle(E0,mu, theta)-dE/2)
    theta11 = get_angle_by_energy(E0, mu, get_energy_by_angle(E0, mu, theta)+dE/2)
    return np.abs(2*np.pi*(np.cos(theta10*np.pi/180)-np.cos(theta11*np.pi/180)))

def get_dBeta(E0:float, theta:float, mu:float, dE:float): 
    """
    method return delta of scattering angle in degrees for 
    specific bin size dE of the analyzer at energy position
    cor__responding to mu==M_target/M_incident
    """
    dEtodB =get_energy_by_angle(E0,mu, theta)*2*np.sin(theta*np.pi/180)
    dEtodB/=np.sqrt(mu**2-(np.sin(theta*np.pi/180))**2)*180/np.pi
    return  dE/dEtodB

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

def get_sensitivity_factor(E0:float, incident_element:str, target_element:str, theta:float, dTheta:float, R : float = -1,  dE: float = 1):
    """
    Returns sensitivity factor for the given incident and target elements.
    The intensity of the peak should be MULTIPLIED by this factor
    if R > 0, dE is considered as R*E1 and the transmission function
    is considered as _a linear function of energy
    """
    mu = get_mass_by_element(target_element)/get_mass_by_element(incident_element)
    if R > 0:
        dE = R*get_energy_by_angle(E0, theta, mu)   
    sensitivity_factor = 1/(get_dBeta(E0, theta, mu, dE)*get_cross_section(incident_element, E0, theta, target_element))
    if R > 0:
        return sensitivity_factor/get_energy_by_angle(E0, theta, mu)   
    else:
        return sensitivity_factor
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
    max_value_mu = 21
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
        if theta>=90:
            min_value_mu = 1+step_mu
        for i_mu in range (int(min_value_mu/step_mu)+1,number_of_points_mu):
            mu = min_value_mu+i_mu*step_mu
            mu_values[i_mu] = mu
            map0[i_mu, i_theta] = get_dBeta(E0, theta, mu, 2)/2
            #print(str(map0[i_mu, i_theta])[0:5], end=" ")
           # file.write(str(map0[i_mu, i_theta])[0:7]+" ")
        #file.write("\n")
        #print("\n")
    #file.close()

    #nipy_spectral   gist_ncar
    plt.figure(figsize=(10, 6))
    plt.contourf(angles,mu_values, map0, cmap='gist_ncar', levels=np.linspace(0.001, 0.35, 200))
    plt.text(80, 0.8, 'restricted zone: μ> sin(θ)', fontsize = 13)
    plt.colorbar(label='Δβ/ΔE, degrees/eV', ticks=[0.001, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
    plt.xlabel('scattering angle θ, degrees', fontsize=12)
    plt.yticks(np.arange(0, max_value_mu, 2))
    plt.ylabel('target atom mass / incident atom mass μ',fontsize=12)
    plt.clim(0.001, 0.35)
    plt.minorticks_on()
    plt.show()

def plot_CrossSection_map(incident_atom, type="scatter"):
    """
    Method to plot the map of dSidma/dOmega values in dependence of scattering angle and mu
    type = "scatter" or "recoil" for scattering or recoils
    """
    global E0
    E0 = 10000
    step_mu = 1
    min_value_mu = 1 if "scatter" in type else 0
    max_value_mu = total_num_elem-10
    number_of_points_mu = int((max_value_mu-min_value_mu)/step_mu)

    incident_mass = get_mass_by_element(incident_atom)
    step_theta = 0.5
    min_value_theta = 10
    max_value_theta = 178
    if "recoil" in type:
        max_value_theta = 88
    number_of_points_theta = int((max_value_theta-min_value_theta)/step_theta)
    
    map0 = np.zeros((number_of_points_mu, number_of_points_theta))
    angles = np.zeros(number_of_points_theta)
    mu_values = np.zeros(number_of_points_mu)

    for i_theta in range (0,number_of_points_theta):
        theta = min_value_theta+i_theta*step_theta
        angles[i_theta] = theta
        # for escaping zone restricted by kinematics laws
        if theta<90:
            min_value_mu_string = get_element_by_mass(np.sin(theta*np.pi/180)*incident_mass)
        else:
            min_value_mu_string= incident_atom
        if "scatter" in type:
            min_value_mu = next (i for i in range(1, total_num_elem) if get_element_info_by_atomic_number(i)[1] == min_value_mu_string)
        for i_mu in range (int(min_value_mu/step_mu)+1,number_of_points_mu):
            mu = min_value_mu+i_mu*step_mu
            mu_values[i_mu] = get_element_info_by_atomic_number(int(mu))[0]
            map0[i_mu, i_theta] = (get_cross_section(incident_atom, E0, theta,get_element_info_by_atomic_number(int(mu))[1],  type))  #get_dBeta(E0, theta, mu, 2)/2
            #print(str(map0[i_mu, i_theta])[0:5], end=" ")
           # file.write(str(map0[i_mu, i_theta])[0:7]+" ")
        #file.write("\n")
        #print("\n")
    #file.close()

    plt.figure(figsize=(10, 6))
    restricted_zone = ('restricted zone: ' if language=='eng' else 'запрещённая зона ')+'μ> sin(θ)' 
    if "recoil" in type:
        start_log = -3
        end_log = 1
        ticks=[1E-3, 1E-2, 1E-1, 1E0, 1E1]
        start_y_tick = 0
        end_y_tick = 201
    else:
        start_log = -3.2
        end_log = 1
        ticks=[1E-4, 1E-3, 1E-2, 1E-1, 1E0, 1E1]
        start_y_tick = 10
        end_y_tick = 210
        plt.text(60, 20, restricted_zone, fontsize = 13)
    plt.contourf(angles, mu_values, map0, cmap='gist_ncar', levels=np.logspace(start_log, end_log, 300), norm=LogNorm())
    plt.yticks(np.arange(start_y_tick, end_y_tick, 10))
    plt.ylim(start_y_tick, end_y_tick)
    if language=='eng':
        plt.ylabel('target atom mass, a.m.u.',fontsize=12)
        plt.xlabel('scattering angle θ, degrees', fontsize=12)
        plt.colorbar(label=f'cross-section for {incident_atom} incident atom, Å2/sr', ticks=ticks)

    else:
        plt.ylabel('масса атома мишени, а.е.м.',fontsize=12)
        if "scatter" in type:
            plt.xlabel('угол рассеяния θ, градусы', fontsize=12)
            plt.colorbar(label=f'сечение для налетающего {incident_atom}, Å2/ср', ticks=ticks)
        else:
            plt.xlabel('угол выбивания θ, градусы', fontsize=12)
            plt.colorbar(label=f'сечение атома, выбитого {incident_atom}, Å2/ср', ticks=ticks)
    #plt.clim(0.001, 0.35)
    if "scatter" in type:
        plt.title(f'Scattering cross-section map for {incident_atom} incident ion', fontsize=14)
    else:
        plt.title(f'Recoil cross-section map for {incident_atom} incident ion', fontsize=14)
    plt.minorticks_on()
    plt.show()

def plot_spectrum_with_concs(spectrum: spectrum, title = None):
    """
    Method to plot the spectrum with quantified peaks
    """
    plt.figure(figsize=(10, 6))
    plt.plot(spectrum.spectrum_en[int(Emin/spectrum.step):]/1000, spectrum.spectrum_int[int(Emin/spectrum.step):], '-', linewidth=2, label=spectrum.calc_name) 
    plt.plot(spectrum.spectrum_en[spectrum.peaks]/1000, spectrum.spectrum_int[spectrum.peaks], "o", color='red')

    i=0
    for x,y in zip(spectrum.spectrum_en[spectrum.peaks]/1000,spectrum.spectrum_int[spectrum.peaks]):

        label = str(spectrum.target_components[i])+"\n"+str(spectrum.elem_conc_by_Icorr[i])[0:4]+" %"
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

def _young( E, E0, A, R, FWHM, B, K):
    #R=1
    I_el = A*np.exp((-(1-R)*2.77259*(E-E0)/(FWHM+0.001)*2))/(R*(E-E0)**2+((FWHM+0.001)/2)**2)
    # UPDATE 250520 150 is to reduce inelactic compared to elastic at peak position
    I_inel = B*(np.pi-2*np.arctan(2*(E-E0+150)/(FWHM+0.001)))
    I_tail = np.exp(-K/(np.sqrt(E)+0.001))    
    return I_el+I_inel*I_tail

class fitted_spectrum:
    
    __param_bounds = ([0,0,0,0,0,0,0,0],[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])
    
    def __init__(self, spectrum:spectrum, target_element1:str, target_element2:str):
        """
        Constructor of the fitted spectrum class.
        """
        self.__spectrum = spectrum
        self.__target_element1 = target_element1
        self.__target_element2 = target_element2
        
        try:
            self.__spectrum.do_elemental_analysis()
        except:
            print("ERROR in fitted_spectrum constructor: spectrum is not analyzed")
        
        # for better fitting real peak positions are better than calculated ones
        # due to some small inelastic shifts
        # Additioanlly, we shift the position of the first peak to the left by 20 eV
        # as it is located on the left slope of the more energetic one
        #self.__E01 = spectrum.peaks[0]*spectrum.step-20
        #self.__E02 = spectrum.peaks[1]*spectrum.step
        
        # TODO refactor !
        # Yet another way to find peaks
        
        self.__E01 = get_energy_by_angle(spectrum.E0,  get_mass_by_element(target_element1)/spectrum.M0, spectrum.scattering_angle)
        self.__E02 = get_energy_by_angle(spectrum.E0,  get_mass_by_element(target_element2)/spectrum.M0, spectrum.scattering_angle)
        
        self.__E01 = int((self.__E01-100))+peak_pos(spectrum.spectrum_int[int((self.__E01-100)/spectrum.step):int((self.__E01+100)/spectrum.step)])*spectrum.step-20
        self.__E02 = int((self.__E02-100))+peak_pos(spectrum.spectrum_int[int((self.__E02-100)/spectrum.step):int((self.__E02+100)/spectrum.step)])*spectrum.step

        pars, covariance = curve_fit(self.__twoCompTargetFitting, 
                                     spectrum.spectrum_en, spectrum.spectrum_int, 
                                     method = 'dogbox', maxfev=500000, bounds=self.__param_bounds, loss='linear', ftol=1E-9)

        self.__pars = pars
        
        
    def __twoCompTargetFitting(self, E, A1, FWHM1, B1, K1, A2, FWHM2, B2, K2):

        # We specify R=1 to use Lorenzian distribution for elastic part of the spectrum
        return _young(E, self.__E01, A1, 1, FWHM1, B1, K1) + _young(E, self.__E02, A2, 1, FWHM2, B2, K2)


    def get_fitted_spectrum(self):
        """
        Method to get the fitted spectrum
        """
        fitted_spectrum_int = self.__twoCompTargetFitting(self.__spectrum.spectrum_en, self.__pars[0], self.__pars[1], self.__pars[2], self.__pars[3], self.__pars[4], self.__pars[5], self.__pars[6], self.__pars[7])
        return norm(fitted_spectrum_int)
    
    
    def get_elastic_part(self, element: str):
        """
        Method to get the elastic part of the fitted spectrum
        """
        
        if element not in self.__target_element1 and element not in self.__target_element2:
            print("ERROR in get_elastic_part: element not found")
            return None
        if element in self.__target_element1:
            peak_position = self.__E01
            elastic_part_int = [_young(E, peak_position, self.__pars[0], 1, self.__pars[1], 0, 0) for E in self.__spectrum.spectrum_en]
            return elastic_part_int
        if element in self.__target_element2:
            peak_position = self.__E02
            elastic_part_int = [_young(E, peak_position, self.__pars[4], 1, self.__pars[5], 0, 0) for E in self.__spectrum.spectrum_en]
            return elastic_part_int
    
    def get_inelastic_part(self, element: str):
        """
        Method to get the inelastic part of the fitted spectrum
        """
        if element not in self.__target_element1 and element not in self.__target_element2:
            print("ERROR in get_elastic_part: element not found")
            return None
        if element in self.__target_element1:
            peak_position = self.__E01
            inelastic_part_int = [_young(E, peak_position, 0, 1, self.__pars[1], self.__pars[2], self.__pars[3]) for E in self.__spectrum.spectrum_en]   
            return inelastic_part_int
        if element in self.__target_element2:
            peak_position = self.__E02
            inelastic_part_int = [_young(E, peak_position, 0, 1, self.__pars[5], self.__pars[6], self.__pars[7]) for E in self.__spectrum.spectrum_en]
            return inelastic_part_int

    def get_concentration(self):
        """
        Method to get the concentration of the more heavy element in the fitted spectrum
        """
        int1 = sum(self.get_elastic_part(self.__target_element1))/get_cross_section(self.__spectrum.incident_atom, 
                                                                                    self.__spectrum.E0, 
                                                                                    self.__spectrum.scattering_angle, 
                                                                                    self.__target_element1)
        int2 = sum(self.get_elastic_part(self.__target_element2))/get_cross_section(self.__spectrum.incident_atom, 
                                                                                    self.__spectrum.E0, 
                                                                                    self.__spectrum.scattering_angle, 
                                                                                    self.__target_element2)

        return int2/(int1+int2)*100
        
    def get_conc_by_inten(self, R):
        int1 = max(self.get_elastic_part(self.__target_element1))*get_sensitivity_factor(self.__spectrum.E0, self.__spectrum.incident_atom,
                                                                                         self.__target_element1, self.__spectrum.scattering_angle,
                                                                                         self.__spectrum.dTheta, R)
        int2 = max(self.get_elastic_part(self.__target_element2))*get_sensitivity_factor(self.__spectrum.E0, self.__spectrum.incident_atom,
                                                                                         self.__target_element2, self.__spectrum.scattering_angle,
                                                                                         self.__spectrum.dTheta, R)
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


# Taken from LEIS_calculator made by Ivan Nikitin
# see https://elibrary.ru/item.asp?id=54049055 for details
# Global variables
__m = [0, 0]  # Masses of projectile and target
__z = [0, 0]  # Atomic numbers of projectile and target

__chosen_potential = 1 # 0 - TFM, 1 - ZBL, 2 - KRC
__EnergyCMS = 0 # Energy in the center of mass system
__screeningLength = 0 # Screening length
__potentialEnergy = 0 # Potential energy
__curv = 0 # Curvature
__average_Z = 0 # Average atomic number
__impactParameterDimentionless = 0 
__scatteringAngleCMS = 0 # Scattering angle in the center of mass system 
__energyReduced = 0 # Reduced energy
__closestApproachDimentionless = 0 # Closest approach

__C1, __C2, __C3, __C4, __C5 = 0, 0, 0, 0, 0  # Constants for potential
__s1, __s2, __s3, __s4 = 0, 0, 0, 0  # Screening function parameters
__d1, __d2, __d3, __d4 = 0, 0, 0, 0  # Screening function parameters

def __element(ta, por):
    global __m, __z    
    __z[por] = next((z for z in range (1, total_num_elem) 
                   if get_element_info_by_atomic_number(z)[1] == ta), None) #projectile atomic number
        
    __m[por]=next((get_element_info_by_atomic_number(z)[0] for z in range (1, total_num_elem) 
                 if get_element_info_by_atomic_number(z)[1] == ta), None) #projectile atomic mass

def __potential():
    global __potentialEnergy, __average_Z, __screeningLength, __closestApproachDimentionless
    __potentialEnergy = __average_Z * 23.0707 * math.pow(10, -20) * (
        __s1 * math.exp(-__d1 * __closestApproachDimentionless) + 
        __s2 * math.exp(-__d2 * __closestApproachDimentionless) + 
        __s3 * math.exp(-__d3 * __closestApproachDimentionless) + 
        __s4 * math.exp(-__d4 * __closestApproachDimentionless)
    ) / (__closestApproachDimentionless * __screeningLength)

def __impact_parameter(r0):
    global __impactParameterDimentionless, __screeningLength, __EnergyCMS, __potentialEnergy
    __potential()
    if (1 - __potentialEnergy / __EnergyCMS) < 0:
        return 1
    else:
        impact_param = r0 * math.pow(1 - __potentialEnergy / __EnergyCMS, 0.5)
        __impactParameterDimentionless = impact_param / __screeningLength
        return 0

def __curvature(r0):
    global __curv, __closestApproachDimentionless, __potentialEnergy, __screeningLength, __EnergyCMS
    global __s1, __s2, __s3, __s4, __d1, __d2, __d3, __d4
    __curv = 2 * (__EnergyCMS - __potentialEnergy) * r0 / (
        __screeningLength * __potentialEnergy + 
        __average_Z * 23.0707 * math.pow(10, -20) * (
            __s1 * __d1 * math.exp(-__d1 * __closestApproachDimentionless) + 
            __s2 * __d2 * math.exp(-__d2 * __closestApproachDimentionless) + 
            __s3 * __d3 * math.exp(-__d3 * __closestApproachDimentionless) + 
            __s4 * __d4 * math.exp(-__d4 * __closestApproachDimentionless)
        )
    )

def __res():
    global __impactParameterDimentionless, __closestApproachDimentionless, __curv, __scatteringAngleCMS, __energyReduced
    beta = (__C2 + math.pow(__energyReduced, 0.5)) / (__C3 + math.pow(__energyReduced, 0.5))
    A0 = 2 * (1 + __C1 * math.pow(__energyReduced, -0.5)) * __energyReduced * math.pow(__impactParameterDimentionless, beta)
    G = (__C5 + __energyReduced) / ((__C4 + __energyReduced) * (math.pow(1 + A0 * A0, 0.5) - A0))
    d = (__impactParameterDimentionless + __curv + A0 * (__closestApproachDimentionless - __impactParameterDimentionless) / (1 + G)) / (__closestApproachDimentionless + __curv) - math.cos(__scatteringAngleCMS / 2)
    return d

def __approach(initial_energy):
    global __EnergyCMS, __screeningLength, __energyReduced, __average_Z, __closestApproachDimentionless
    
    __EnergyCMS = 1.6021766 * math.pow(10, -12) * initial_energy / (1 + __m[0] / __m[1])
    z0 = math.pow(__z[0], 0.5) + math.pow(__z[1], 0.5)
    __screeningLength = 0.8853 * 0.529 * math.pow(10, -8) / math.pow(z0, 0.666666666)
    
    if __chosen_potential == 1 or __chosen_potential == 2:
        z0 = math.pow(__z[0], 0.23) + math.pow(__z[1], 0.23)
        __screeningLength = 0.88534 * 0.529 * math.pow(10, -8) / z0
        
    __energyReduced = __screeningLength * __EnergyCMS / (__z[0] * __z[1] * 23.0707 * math.pow(10, -20))
    __average_Z = __z[0] * __z[1]
    
    closest_approach_left = 0
    closest_approach_right = 5 * math.pow(10, -8)
    
    for _ in range(40):
        closest_approach_average = (closest_approach_left + closest_approach_right) / 2
        __closestApproachDimentionless = closest_approach_average / __screeningLength
        q = __impact_parameter(closest_approach_average)
        
        if q == 0:
            __curvature(closest_approach_average)
            solutions_difference = __res()
            if solutions_difference > 0:
                closest_approach_right = closest_approach_average
            else:
                closest_approach_left = closest_approach_average
        if q == 1:
            closest_approach_left = closest_approach_average
            
    closest_approach_average = (closest_approach_left + closest_approach_right) / 2
    return closest_approach_average


def __recoiled(initial_energy, recoiled_angle):
    global __scatteringAngleCMS, __impactParameterDimentionless
    __scatteringAngleCMS = math.pi - 2 * recoiled_angle
    #recoiledEnergy = initial_energy * 4 * __m[0] * __m[1] * math.pow(math.cos(recoiled_angle), 2) / math.pow(__m[0] + __m[1], 2)
    scattered_angle = math.atan(__m[1] * math.sin(__scatteringAngleCMS) / (__m[0] + __m[1] * math.cos(__scatteringAngleCMS)))
    if scattered_angle < 0:
        scattered_angle += math.pi

    if recoiled_angle == 0:
        __impactParameterDimentionless = 0
        
    # Calculate differential cross section
    recoiled_angle_left = recoiled_angle - 0.000001
    __scatteringAngleCMS = math.pi - 2 * recoiled_angle_left
    closest_approach = __approach(initial_energy)
    __impact_parameter(closest_approach)
        
    impact_parameter1 = __impactParameterDimentionless * __screeningLength * math.pow(10, 8)
    recoiled_angle_right = recoiled_angle + 0.000001
    __scatteringAngleCMS = math.pi - 2 * recoiled_angle_right
    closest_approach = __approach(initial_energy)
    __impact_parameter(closest_approach)
    impact_parameter2 = __impactParameterDimentionless * __screeningLength * math.pow(10, 8)

    dsdo = abs((abs(math.pow(impact_parameter1, 2) - math.pow(impact_parameter2, 2))) / (2 * math.sin(recoiled_angle) * 0.000002))
    return dsdo


def __scattered(initial_energy, scattering_angle):
    
    global __scatteringAngleCMS, __m, __z, __impactParameterDimentionless, __screeningLength
    scattering_angle_left = scattering_angle - 0.00001
    scattering_angle_right = scattering_angle + 0.00001
    
    # Calculate energy for left angle
    energy_scattered = initial_energy * math.pow(
        (math.cos(scattering_angle_left) + math.pow(math.pow(__m[1] / __m[0], 2) - math.pow(math.sin(scattering_angle_left), 2), 0.5)) / (1 + __m[1] / __m[0]), 2
    )
    # Calculate CMS angle for left
    __scatteringAngleCMS = math.atan(
        math.sin(scattering_angle_left) * math.pow(2 * __m[0] * energy_scattered, 0.5) / 
        ((__m[0] * (math.cos(scattering_angle_left) * math.pow(2 * __m[0] * energy_scattered, 0.5) / __m[0] - 
        math.pow(2 * __m[0] * initial_energy, 0.5) / (__m[0] + __m[1]))))
    )
    
    if __scatteringAngleCMS < 0:
        __scatteringAngleCMS = math.pi + __scatteringAngleCMS
    
    closest_approach = __approach(initial_energy)
    __impact_parameter(closest_approach)
    impact_parameter1 = __impactParameterDimentionless * __screeningLength * math.pow(10, 8)
    
    # Calculate for right angle
    energy_scattered = initial_energy * math.pow(
        (math.cos(scattering_angle_right) + math.pow(math.pow(__m[1] / __m[0], 2) - math.pow(math.sin(scattering_angle_right), 2), 0.5)) / (1 + __m[1] / __m[0]), 2
    )
    # Calculate CMS angle for right
    __scatteringAngleCMS = math.atan(
        math.sin(scattering_angle_right) * math.pow(2 * __m[0] * energy_scattered, 0.5) / 
        ((__m[0] * (math.cos(scattering_angle_right) * math.pow(2 * __m[0] * energy_scattered, 0.5) / __m[0] - 
        math.pow(2 * __m[0] * initial_energy, 0.5) / (__m[0] + __m[1]))))
    )
    
    if __scatteringAngleCMS < 0:
        __scatteringAngleCMS = math.pi + __scatteringAngleCMS
    
    closest_approach = __approach(initial_energy)
    __impact_parameter(closest_approach)
    impact_parameter2 = __impactParameterDimentionless * __screeningLength * math.pow(10, 8)
    
    # Calculate differential cross section
    dsdo = ((impact_parameter1 - impact_parameter2) * (impact_parameter1 + impact_parameter2) / 
           (2 * math.sin(scattering_angle) * 2E-5))
    return dsdo

def set_potential(pot:str):
    """
    Method to set the potential type: ZBL, TFM, KRC
    """
    pots = ("TFM", "ZBL", "KRC")
    global __chosen_potential
    try:
        __chosen_potential = pots.index(pot)
    except:
        print("ERROR in set_potential: potential not found")

def get_cross_section(incident_symbol, E0, o1, target_symbol, type="scatter"):
    
    """
    returns cross section of the scattering process for given incident and target elements
    incident_symbol - symbol of the incident element
    E0 - energy of the incident particle
    o1 - scattering angle
    target_symbol - symbol of the target element  
    defaul potential is ZBL. Use set_potential(pot) to change it
    type = "scatter" or "recoil" for scattering or recoils
    """    
    global __chosen_potential, __C1, __C2, __C3, __C4, __C5, __s1, __s2, __s3, __s4, __d1, __d2, __d3, __d4
    
    o2 = o1 * math.pi / 180

    #print (incident_symbol+" -> "+target_symbol)
    __element(incident_symbol, 0)  # Example: Neon as projectile
    __element(target_symbol, 1)  # Example: Tungsten as target
    
    if __chosen_potential == 0:
        __s1, __s2, __s3, __s4 = 0.35, 0.55, 0.1, 0
        __d1, __d2, __d3, __d4 = 0.3, 1.2, 6, 0
        __C1, __C2, __C3, __C4, __C5 = 0.6743, 0.009611, 0.005175, 6.314, 10
    elif __chosen_potential == 1:
        __s1, __s2, __s3, __s4 = 0.028171, 0.28022, 0.50986, 0.18175
        __d1, __d2, __d3, __d4 = 0.20162, 0.40290, 0.94229, 3.1998
        __C1, __C2, __C3, __C4, __C5 = 0.99229, 0.011615, 0.0071222, 9.3066, 14.813
    elif __chosen_potential == 2:
        __s1, __s2, __s3, __s4 = 0.190945, 0.473674, 0.335381, 0
        __d1, __d2, __d3, __d4 = 0.278544, 0.637174, 1.919249, 0
        __C1, __C2, __C3, __C4, __C5 = 1.0144, 0.235809, 0.126, 6.9350, 8.3550
    
    if "scatter" in type :  
        if  __m[0] > __m[1]: 
            try:         
                if o2 >= np.arcsin(__m[1]/__m[0]):
                    print(f"ERROR: scattering angle should be lower than {np.arcsin(__m[1]/__m[0])*180/np.pi:.2f} degrees. Now it is {o1} degrees")
                    return 0
                else:
                    return __scattered(E0, o2)
            except ValueError:
                print(ValueError)
                print("This error has not been expected ha-ha")
                return 0
        else:
            return __scattered(E0, o2)
    elif "recoil" in type:
        if o1>=90:
            print(f"ERROR: recoil angle should be less than 90 degrees. Now it is {o1} degrees")
            return 0
        else:
            return __recoiled(E0, o2)

