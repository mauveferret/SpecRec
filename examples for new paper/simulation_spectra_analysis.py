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


import os,  sys
# changing working directoru to the SpecRec dir
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(parent_dir) 

import numpy as np
import matplotlib.pyplot as plt

BASE_LIB_PATH = "tools"
sys.path.insert(1, os.getcwd()+os.sep+BASE_LIB_PATH)
import LEIS_tools as leis
import spectraConvDeconv_tools as SCD

spectrum_path0 = os.getcwd()+os.sep+"raw_data"+os.sep
dE=SCD.step

#####################################    PRESETS      #####################################

#spectrum_path +="sim_Ne6keV140deg_BaCoGd.dat"
#spectrum_path +="sim_Ne15keV32deg0.9dBeta_AuPdthin.dat"
spectrum_path0 +="temp"#+os.sep+"sim_Ne11.0keV45.0deg1.0_W30Cr70.dat"

#############################################################################################

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
                        
            # TODO
            data = leis.spectrum(spectrum_path0+os.sep+calc, 10)
            data.do_elemental_analysis()
            
            conc_I = data.elem_conc_by_I[-1]
            conc_S = data.elem_conc_by_S[-1]
            conc_corrI = data.elem_conc_by_corrI[-1]
            
            #print(str(data.calc_name)+" "+str(data.spectrum_en[data.peaks[-1]])+" eV "+str(data.target_masses[i])[0:5]+" a.m.u. "
            #  +str(target_components[i])+" "+str(dBetas[i])[0:5]+" deg "+str(dEs[i])[0:5]+" eV "+str(cross_sections[i])[0:4]+" A2/sr")
            
            if conc_I !=0:
                concs_I.append(conc_I)
                concs_S.append(conc_S)
                concs_Icorr.append(conc_corrI)
                
            plt.plot(id, conc_I, "x", label = data.calc_name)
            plt.plot(id, conc_S, "o", label = data.calc_name)
            plt.plot(id, conc_corrI, "*", label = data.calc_name)

        
    plt.xlabel('energy, keV', fontsize=12)
    plt.ylabel('intensity, norm.',fontsize=12)
    #plt.title("Energy spectra of "+spectrum_path[:-4], y=1.01, fontsize=10)
    plt.minorticks_on()
    plt.legend( frameon=False, loc='lower right', fontsize=11)
    plt.show()   
    
    print("$$$$$$$$$$$$$$$$$$$$$$$")
    average, dev =  leis.get_standart_deviation(concs_I)
    print("concI     = "+str(average)[0:5]+"±"+str(dev)[0:4])
    average, dev =  leis.get_standart_deviation(concs_S)
    print("concS     = "+str(average)[0:5]+"±"+str(dev)[0:4])
    average, dev =  leis.get_standart_deviation(concs_Icorr)
    print("concIcorr = "+str(average)[0:5]+"±"+str(dev)[0:4])  
    print("$$$$$$$$$$$$$$$$$$$$$$$")
