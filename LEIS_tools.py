"""
This package can be used to postprocess experimental and simulated LEIS spectra. It allows 
to consider real solid angles of elements, determine peak posistions and corresponding elements,
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
from mendeleev import get_all_elements

incident_atom = "Ne"
target_atom1 = "Pd"
target_atom2 = "Au"

E0=15000.0 #projectile enegy, eV
theta=32.0 #scattering angle, deg

e=1.6*10**(-19) #elementary charge, C
k=9*10**9 #Coulomb constant, N*m^2*C^(-2)
a_0=5.29*10**(-11) #Bohr radius, m


elements = get_all_elements()

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
    
def get_angle_by_energy (E1, mu):
    try:
        theta1 = 180/np.pi*np.arccos(0.5*np.sqrt(E1/E0)*(E0/E1*(1-mu)+mu+1))
    except:
        print("Error in get_angle_by_energy")
    return theta1

def get_angle_by_energy2 (E1, mu):
    try: # scatter
        theta1 = 180/np.pi*np.arccos((M0+M1)*(E1 +E0*(M0**2-M1**2)/(M0+M1)**2)/(2*np.sqrt(E0 * E1)*M0))   
    except: # recoil
        theta1 = (90 - 180 * np.arcsin((M0 + M1) * np.sqrt(E1/(E0*M0*M1))/2)/np.pi)
    return theta1

def get_energy_by_angle(theta, mu):
    E1 = E0*((np.cos(theta*np.pi/180)+np.sqrt(mu**2-(np.sin(theta*np.pi/180))**2))/(1+mu))**2
    return E1


def get_target_mass_by_energy(theta, E1):
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
    

def get_element_by_mass(mass):
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

def get_dSigma(theta, mu, dE):
    """
    method return dOmega of scatted particles in steradians for 
    specific bin size dE of the analyzer at energy position
    corresponding to mu==M_target/M_incident
    """
    theta10 = get_angle_by_energy(get_energy_by_angle(theta, mu)-dE/2, mu)
    theta11 = get_angle_by_energy(get_energy_by_angle(theta, mu)+dE/2, mu)
    return np.abs(2*np.pi*(np.cos(theta10*np.pi/180)-np.cos(theta11*np.pi/180)))

def get_dBeta(theta, mu, dE): 
    """
    method return delta of scattering angle in degrees for 
    specific bin size dE of the analyzer at energy position
    corresponding to mu==M_target/M_incident
    """
    return  dE/(get_energy_by_angle(theta, mu)*2*np.sin(theta*np.pi/180))*np.sqrt(mu**2-(np.sin(theta*np.pi/180))**2)*180/np.pi

def get_cross_section(theta):
    """
    Is not finished yet
    """
    set_elements_params()
    theta_r=(theta*np.pi)/180 # theta in radians
    a_U=0.8853*a_0/(Z0**(0.23)+Z1**(0.23)) #ZBL screening length, m
    C=(a_U*M1*E0)/(k*Z0*Z1*e*(M0+M1)) #dimensionless energy
    hi=theta_r+np.arcsin((np.sin(theta_r))*Z0/Z1)
    t=(C**2)*(np.sin(hi/2)**2) #some Lindhard approximation
    ds_dt=np.pi*0.36*(a_U**2)/(2*t**(3/2)) #differential cross-section, m^2
    #print ('ds_dt=', ds_dt/10**(-4))
    return ds_dt/10**(-20)

def get_intensity_correction_OLD(theta = theta):
    dSigma_atom1 = get_dSigma(theta, mu1,1)
    dSigma_atom2 = get_dSigma(theta, mu2,1)
    return dSigma_atom1/dSigma_atom2

def get_intensity_corrections(mu1, mu2, theta = theta, R = -1):
    """
    Calculates intensities correction coefficients for considering
    effect of different collecting angles due to spectra discretization
    To recover true intensities multiply intensity of element with
    peak of highers energy to this coefficient
    
    To take into account real spectrometer geometry with dE/E=const 
    specify any positive R value (R = dE/E).
    """  
    if R <= 0 :
        dBeta2 = get_energy_by_angle(theta, mu2)*2*np.sin(theta*np.pi/180)/np.sqrt(mu2**2-(np.sin(theta*np.pi/180))**2)
        dBeta1 = get_energy_by_angle(theta, mu1)*2*np.sin(theta*np.pi/180)/np.sqrt(mu1**2-(np.sin(theta*np.pi/180))**2)
    if R > 0 :
        dBeta2 = (get_energy_by_angle(theta, mu2))**2*2*np.sin(theta*np.pi/180)/np.sqrt(mu2**2-(np.sin(theta*np.pi/180))**2)
        dBeta1 = (get_energy_by_angle(theta, mu1))**2*2*np.sin(theta*np.pi/180)/np.sqrt(mu1**2-(np.sin(theta*np.pi/180))**2)
    return dBeta2/dBeta1


