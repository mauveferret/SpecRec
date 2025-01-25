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

import  re
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import LEIS_tools as leis
import spectraConvDeconv_tools as SCD

spectrum_path = os.getcwd()+os.sep+"raw_data"+os.sep
#####################################    PRESETS      #####################################



#spectrum_path +="sim_Ne6keV140deg_BaCoGd.dat"
spectrum_path +="sim_Ne15keV32deg0.9dBeta_AuPdthin.dat"

dE=0.9

#####################################    IMPORT DATA      #####################################
SCD.calc_name = spectrum_path.split(os.sep)[-1].split(".")[0]
SCD.Emin = 500
SCD.filter_window_length = 10
SCD.doInputSmooth = True
spectrum_en, spectrum_int = SCD.import_data(spectrum_path)

# get initial params from the filename
leis.incident_atom = re.sub(r'\d', '', spectrum_path.split(os.sep)[-1].split("_")[1].split("keV")[0])
leis.E0 = int(spectrum_path.split(os.sep)[-1].split("_")[1].split("keV")[0].split(leis.incident_atom)[1])*1000
theta = int(spectrum_path.split(os.sep)[-1].split("_")[1].split("keV")[1].split("deg")[0])
leis.set_elements_params()
#####################################    DO PEAKS ANALYSIS    #####################################


peaks, _ = find_peaks(spectrum_int, height=0.3, width=20)
target_masses = [leis.get_target_mass_by_energy(theta, spectrum_en[peaks[i]]) for i in range(len(peaks))]
target_components = [leis.get_element_by_mass(mass) for mass in target_masses]
dBetas = [leis.get_dBeta(theta, mass/leis.M0, dE) for mass in target_masses]

cross_sections = [leis.get_cross_section(leis.incident_atom,leis.E0, theta,1, component) for component in target_components]


E0 = 15000  # Example: 6000 eV
o1 = 140  # Example: 32 degrees
od = 2  # Example: 2 degrees

for i in range(len(peaks)): 
    print(str(spectrum_en[peaks[i]])+" eV "+str(target_masses[i])[0:5]+" a.m.u. "+str(target_components[i])+" "+str(dBetas[i])[0:5]+" deg "+str(cross_sections[i])[0:4]+" A2/sr")

if (len(peaks) == 2):
    int1 = spectrum_int[peaks[0]]/(cross_sections[0]*dBetas[0]*spectrum_en[peaks[0]])
    int2 = spectrum_int[peaks[1]]/(cross_sections[1]*dBetas[1]*spectrum_en[peaks[1]])
    print ("Conc = "+str(int1/(int1+int2)*100)[0:4]+" %")

#####################################    PLOT DATA      #####################################

plt.plot(spectrum_en[SCD.Emin:]/1000, spectrum_int[SCD.Emin:], '-', linewidth=2, label=SCD.calc_name) 
plt.plot(spectrum_en[peaks]/1000, spectrum_int[peaks], "x")

i=0
for x,y in zip(spectrum_en[peaks]/1000,spectrum_int[peaks]):

    label = leis.get_element_by_mass( leis.get_target_mass_by_energy(theta, spectrum_en[peaks[i]]))
    i+=1
    plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(10,0), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('energy, keV', fontsize=12)
plt.ylabel('intensity, norm.',fontsize=12)
#plt.title("Energy spectra of "+spectrum_path[:-4], y=1.01, fontsize=10)
plt.minorticks_on()
plt.legend( frameon=False, loc='lower right', fontsize=11)
plt.show()