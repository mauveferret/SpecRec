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

#####################################    PRESETS      #####################################
spectrum_path0 = os.getcwd()+os.sep+"raw_data"+os.sep+"sim_AuPd"
dE=SCD.step
#############################################################################################

fig, axs = plt.subplots(3, 3) # rows columns

calcs = os.listdir(spectrum_path0)
incident_atoms = ("Ne", "Ar", "Kr")
targets = ("Au30Pd70", "Au50Pd50", "Au70Pd30")

for calc in calcs:
    data = leis.spectrum(spectrum_path0+os.sep+calc)
    data.do_elemental_analysis()
    
    conc_I = data.elem_conc_by_I[-1]
    conc_S = data.elem_conc_by_S[-1]
    conc_corrI = data.elem_conc_by_corrI[-1]

    row = targets.index(calc.split("_")[-1].split(".")[0])
    column = incident_atoms.index(data.incident_atom)
    axs[row , column].plot(data.scattering_angle, conc_I, "x", color="blue")
    axs[row , column].plot(data.scattering_angle, conc_S, "o", color="black")
    axs[row , column].plot(data.scattering_angle, conc_corrI, "*", color="red")

plt.show()


exit(0)

for calc in calcs:
    concs_I = []
    concs_S = []
    concs_Icorr = []
    id = []
    for angle in angles:
        for incident_atom in incident_atoms:
            for target in targets:
                id.append(str(angle)+incident_atom+target)
                data = leis.spectrum(spectrum_path0+os.sep+calc, angle, incident_atom, target)
                data.do_elemental_analysis()
                
                conc_I = data.elem_conc_by_I[-1]
                conc_S = data.elem_conc_by_S[-1]
                conc_corrI = data.elem_conc_by_corrI[-1]
                
                if conc_I !=0:
                    concs_I.append(conc_I)
                    concs_S.append(conc_S)
                    concs_Icorr.append(conc_corrI)
                    
    axs[0, 0].plot(id, concs_I, "x", label = data.calc_name)
    axs[0, 1].plot(id, concs_S, "o", label = data.calc_name)
    axs[0, 2].plot(id, concs_Icorr, "*", label = data.calc_name)
    
    axs[1, 0].plot(id, concs_I, "x", label = data.calc_name)
    axs[1, 1].plot(id, concs_S, "o", label = data.calc_name)
    axs[1, 2].plot(id, concs_Icorr, "*", label = data.calc_name)
    
    axs[2, 0].plot(id, concs_I, "x", label = data.calc_name)
    axs[2, 1].plot(id, concs_S, "o", label = data.calc_name)
    axs[2, 2].plot(id, concs_Icorr, "*", label = data.calc_name)


"""
    fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(x, y)
axs[0, 0].set_title('Axis [0, 0]')
axs[0, 1].plot(x, y, 'tab:orange')
axs[0, 1].set_title('Axis [0, 1]')
axs[1, 0].plot(x, -y, 'tab:green')
axs[1, 0].set_title('Axis [1, 0]')
axs[1, 1].plot(x, -y, 'tab:red')
axs[1, 1].set_title('Axis [1, 1]')

for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
    
"""



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


