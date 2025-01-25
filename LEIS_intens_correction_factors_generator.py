"""
This program allows to provide quantification of the element analysis by LEIS

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

If you have questions regarding this program, please contact NEEfimov@mephi.ru
"""

import os
# changing working directoru to the location of py file
os.chdir(os.path.dirname(os.path.realpath(__file__))) 

import matplotlib
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import LEIS_tools as leis
import spectraConvDeconv_tools as SCD


leis.set_elements_params()

step_mu = 0.001
min_value_mu = 0.5
max_value_mu = 10
number_of_points_mu = int((max_value_mu-min_value_mu)/step_mu)

step_theta = 10 #0.5
min_value_theta = 10
max_value_theta = 170
number_of_points_theta = int((max_value_theta-min_value_theta)/step_theta)

map0 = np.zeros((number_of_points_mu, number_of_points_theta))
values0 = np.zeros(number_of_points_theta)
values1 = np.zeros(number_of_points_theta)
values2 = np.zeros(number_of_points_theta)
angles = np.zeros(number_of_points_theta)
mu_values = np.zeros(number_of_points_mu)

target_atom1 = "Pd"
target_atom2 = "Au"
leis.set_elements_params()
mu_v01 = leis.mu1
mu_v02 = leis.mu2

target_atom1 = "Cr"
target_atom2 = "W"
leis.set_elements_params()
mu_v03 = leis.mu1
mu_v04 = leis.mu2

target_atom1 = "Se"
target_atom2 = "Bi"
leis.set_elements_params()
mu_v05 = 0.5
mu_v06 = 2

spectrum_path = os.getcwd()+os.sep+"raw_data"+os.sep
spectrum_path += "ex1_sim_Ne6kev140deg_GdBaCo"+os.sep+"Gd20Ba20Co60.dat"
SCD.calc_name = spectrum_path.split(os.sep)[-1].split(".")[0]
SCD.Emin = 1000
spectrum_en, spectrum_int = SCD.import_data(spectrum_path)


# СДЕЛАТЬ РАСЧЁТ ЭНЕРГИЙ И УГЛОВ ВЫБИВАНИЯ!

for i_theta in range (0,number_of_points_theta):
    theta = min_value_theta+i_theta*step_theta 

    values0[i_theta] = leis.get_simple_intensity_corrections(mu_v01, mu_v02, theta)
    values1[i_theta] = leis.get_simple_intensity_corrections(mu_v03, mu_v04, theta)
    values2[i_theta] = leis.get_simple_intensity_corrections_real_spectrometer(mu_v03, mu_v04, theta)
    
    a = leis.get_angle_by_energy(leis.get_energy_by_angle(theta,mu_v02)-10,mu_v02)-leis.get_angle_by_energy(leis.get_energy_by_angle(theta,mu_v02)+10,mu_v02)
    b = 20/(leis.get_energy_by_angle(theta, mu_v02)*2*np.sin(theta*np.pi/180))*np.sqrt(mu_v02**2-(np.sin(theta*np.pi/180))**2)*180/np.pi
    print(str(theta)+" "+str(a/b)+" "+str(a)+" "+str(b))
    
    angles[i_theta] = theta
    
plt.plot(angles, values0, 'o-', linewidth=2, label="Ne -> AuPd 15 keV") 
plt.plot(angles, values1, 'o-', linewidth=2, label="Ne -> WPd 15 keV") 
plt.plot(angles, values2, 'o-', linewidth=2, label="Ne -> WCr 15 keV") 

plt.xlabel('scattering angle, degrees', fontsize=12)
plt.ylabel('intensity correction coefficient',fontsize=12)
#plt.title("Energy spectra of "+spectrum_path[:-4], y=1.01, fontsize=10)
plt.minorticks_on()
plt.legend( frameon=False, loc='upper left', fontsize=11)
plt.show()

exit(0)

file = open('leis_out.txt', 'w')
for i_theta in range (0,number_of_points_theta):
    theta = min_value_theta+i_theta*step_theta
    angles[i_theta] = theta
    for i_mu in range (int((np.sin(theta*np.pi/180)-min_value_theta)/step_theta)+5,number_of_points_mu):
        mu = min_value_mu+i_mu*step_mu
        mu_values[i_mu] = mu
        map0[i_mu, i_theta] = np.log10(get_dSigma(theta, mu)*1E6)
        #print(str(values0[i_mu, i_theta])[0:5], end=" ")
        file.write(str(map0[i_mu, i_theta])[0:7]+" ")
    file.write("\n")
    #print("\n")
file.close()

#plt.imshow(values0, extent=[min_value_theta, max_value_theta, min_value_mu, max_value_mu], aspect='auto',  norm=matplotlib.colors.LogNorm())    
plt.contourf(angles,mu_values, map0, cmap='winter', levels=np.linspace(1, 4.6, 50))
plt.colorbar(label='log10(dSigma, m^2)', ticks=np.linspace(1, 4.6, 50))
plt.xlabel('угол рассеяния, градусы', fontsize=12)
plt.ylabel('масса атома мишени / масса снаряда',fontsize=12)
plt.clim(1, 4.6)
plt.show()

max_angle = 170
values = np.zeros(int(max_angle/10))
values2 = np.zeros(int(max_angle/10))
values3 = np.zeros(int(max_angle/10))
angles = np.zeros(int(max_angle/10))

for theta in range (10, max_angle+1, 10):
    values[int(theta/10)-1] = get_intensity_correction("Pd", "Au", theta)	
    values2[int(theta/10)-1] = get_intensity_correction("Pd", "W", theta)	
    values3[int(theta/10)-1] = get_intensity_correction("Cr", "W", theta)	
    angles[int(theta/10)-1] = theta 

plt.plot(angles, values, 'o-', linewidth=2, label="Ne -> AuPd 15 keV") 
plt.plot(angles, values2, 'o-', linewidth=2, label="Ne -> WPd 15 keV") 
plt.plot(angles, values3, 'o-', linewidth=2, label="Ne -> WCr 15 keV") 

plt.xlabel('scattering angle, degrees', fontsize=12)
plt.ylabel('intensity correction coefficient',fontsize=12)
#plt.title("Energy spectra of "+spectrum_path[:-4], y=1.01, fontsize=10)
plt.minorticks_on()
plt.legend( frameon=False, loc='upper left', fontsize=11)
plt.show()
