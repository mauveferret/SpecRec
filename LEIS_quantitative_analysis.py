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

spectrum_path0 = os.getcwd()+os.sep+"raw_data"+os.sep
dE=SCD.step


#####################################    PRESETS      #####################################

#spectrum_path +="sim_Ne6keV140deg_BaCoGd.dat"
#spectrum_path +="sim_Ne15keV32deg0.9dBeta_AuPdthin.dat"
spectrum_path0 +="temp"#+os.sep+"sim_Ne11.0keV45.0deg1.0_W30Cr70.dat"

#############################################################################################

def get_standart_deviation(data):
    n = len(data)
    average = sum(data)/n
    standart_deviation = 0
    for value in data:
        standart_deviation+=value**2
    standart_deviation=1/n*np.sqrt(standart_deviation-n*average**2)
    return average, standart_deviation

def get_concentrations(calc_path):
    spectrum_path=spectrum_path0+os.sep+calc_path
    data = leis.spectrum(2.0)
    data.import_data(spectrum_path, 10)
    spectrum_en = data.spectrum_en
    spectrum_int = data.spectrum_int
    
    peaks, _ = find_peaks(spectrum_int, prominence=0.04, width=5, distance=50)
    target_masses = [leis.get_target_mass_by_energy(data.theta, data.M0, data.E0, spectrum_en[peaks[i]]) for i in range(len(peaks))]
    target_components = [leis.get_element_by_mass(mass) for mass in target_masses]

    # mass resolution is not perfect and sometimes LEIS_tools give neighbor elements
    # which is not favorable for quantitative estimations
    for i in range (0,len(target_components)):
        if "Ir" in target_components[i] or "At" in target_components[i]:
            target_components[i]="Au"
            target_masses[i] = leis.get_mass_by_element(target_components[i])
        if "Cd" in target_components[i] or "Ag" in target_components[i]:
            target_components[i]="Pd"
            target_masses[i] = leis.get_mass_by_element(target_components[i])
        if "Ta" in target_components[i] or "Hf" in target_components[i]:
            target_components[i]="W"
            target_masses[i] = leis.get_mass_by_element(target_components[i])
    

    dBetas = [leis.get_dBeta(data.E0, data.theta, mass/data.M0, dE) for mass in target_masses]
    dEs = [leis.get_dE(data.E0, data.theta, mass/data.M0, data.dTheta) for mass in target_masses]
    cross_sections = [leis.get_cross_section(data.incident_atom,data.E0, data.theta, data.dTheta, component) for component in target_components]

    for i in range(len(peaks)): 
        print(str(SCD.calc_name)+" "+str(spectrum_en[peaks[i]])+" eV "+str(target_masses[i])[0:5]+" a.m.u. "+str(target_components[i])+" "+str(dBetas[i])[0:5]+" deg "+str(dEs[i])[0:5]+" eV "+str(cross_sections[i])[0:4]+" A2/sr")


    """
    plt.plot(spectrum_en[int(SCD.Emin/SCD.step):]/1000, spectrum_int[int(SCD.Emin/SCD.step):], '-', linewidth=2, label=SCD.calc_name) 
    plt.plot(spectrum_en[peaks]/1000, spectrum_int[peaks], "x")

    i=0
    for x,y in zip(spectrum_en[peaks]/1000,spectrum_int[peaks]):

        label = target_components[i]
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
    """
    
    #print(spectrum_path.split(os.sep)[-1])
    #print("conc by  Intens   Area   corrIntens, %")
    conc_I = 0
    conc_S = 0
    conc_corrI =0
    if (len(peaks) == 2):
        int1 = spectrum_int[peaks[0]]/(cross_sections[0])
        int2 = spectrum_int[peaks[1]]/(cross_sections[1])
        conc_I = int1/(int1+int2)*100
        
        int1 = sum(spectrum_int[peaks[0]-int(dEs[0]/SCD.step*1.2):peaks[0]+int(dEs[0]/SCD.step*1.2)])/cross_sections[0]
        int2 = sum(spectrum_int[peaks[1]-int(dEs[1]/SCD.step*1.2):peaks[1]+int(dEs[1]/SCD.step*1.2)])/cross_sections[1]
        conc_S = int1/(int1+int2)*100
        
        int1 = spectrum_int[peaks[0]]/(cross_sections[0]*dBetas[0])
        int2 = spectrum_int[peaks[1]]/(cross_sections[1]*dBetas[1])
        conc_corrI = int1/(int1+int2)*100
        
        #print ("Conc = "+str(conc_I)[0:4]+" "+str(conc_S)[0:4]+" "+str(conc_corrI)[0:4]+" ")
        return conc_I, conc_S, conc_corrI, data.calc_name
    print("ERROR in finding peaks!")
    return 0,0,0, 0

#####################################    DO PEAKS ANALYSIS    #####################################


calcs = os.listdir(spectrum_path0)


# causes error due to 50 in angle!!!!
concs = (30, 50, 70)

for conc in concs: 
    
    concs_I = []
    concs_S = []
    concs_Icorr = []
    
    id=0
    for calc in calcs:
    
        if str(conc) in calc[-6:-4] and not "50.0deg" in calc[-6:-4]:  
            id+=1        
            conc_I, conc_S, conc_corrI, calc_name = get_concentrations(calc)
            if conc_I !=0:
                concs_I.append(conc_I)
                concs_S.append(conc_S)
                concs_Icorr.append(conc_corrI)
                
            plt.plot(id, conc_I, "x", label = calc_name)
            plt.plot(id, conc_S, "o", label = calc_name)
            plt.plot(id, conc_corrI, "*", label = calc_name)

        
    plt.xlabel('energy, keV', fontsize=12)
    plt.ylabel('intensity, norm.',fontsize=12)
    #plt.title("Energy spectra of "+spectrum_path[:-4], y=1.01, fontsize=10)
    plt.minorticks_on()
    plt.legend( frameon=False, loc='lower right', fontsize=11)
    plt.show()   
    
    print("$$$$$$$$$$$$$$$$$$$$$$$")
    average, dev =  get_standart_deviation(concs_I)
    print("concI     = "+str(average)[0:5]+"±"+str(dev)[0:4])
    average, dev =  get_standart_deviation(concs_S)
    print("concS     = "+str(average)[0:5]+"±"+str(dev)[0:4])
    average, dev =  get_standart_deviation(concs_Icorr)
    print("concIcorr = "+str(average)[0:5]+"±"+str(dev)[0:4])  
    print("$$$$$$$$$$$$$$$$$$$$$$$")

#####################################    PLOT DATA      #####################################

exit(0)

get_concentrations("sim_Ne15.0keV20.0deg2.0_Au50Pd50.dat")

exit(0)

plt.plot(spectrum_en[int(SCD.Emin/SCD.step):]/1000, spectrum_int[int(SCD.Emin/SCD.step):], '-', linewidth=2, label=SCD.calc_name+"\n"+"conc="+str(conc_corrI)[0:4]) 
plt.plot(spectrum_en[peaks]/1000, spectrum_int[peaks], "x")

i=0
for x,y in zip(spectrum_en[peaks]/1000,spectrum_int[peaks]):

    label = target_components[i]
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