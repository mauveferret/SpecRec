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
import numpy as np
import matplotlib.pyplot as plt
# changing working directoru to the SpecRec dir
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(parent_dir) 

BASE_LIB_PATH = "tools"
sys.path.insert(1, os.getcwd()+os.sep+BASE_LIB_PATH)
import LEIS_tools as leis

#####################################    PRESETS      #####################################
spectrum_path0 = os.getcwd()+os.sep+"raw_data"+os.sep+"sim_AuPd_dE200eVdBeta1deg"
leis.Emin = 10
dE = 2
#############################################################################################

fig, axs = plt.subplots(3, 3, figsize=(20, 20))
plt.subplots_adjust(wspace=0.032, hspace=0.032) # Adjust the width and height space between subplots

calcs = os.listdir(spectrum_path0)
incident_atoms = ("Ne", "Ar", "Kr")
targets = ("Au30Pd70", "Au50Pd50", "Au70Pd30")
datas: list[leis.spectrum] = []

# plot all point 
for calc in calcs:
    data = leis.spectrum(spectrum_path0+os.sep+calc, 50)
    data.do_elemental_analysis()
    datas.append(data)
    
    conc_I = data.elem_conc_by_I[-1]
    conc_S = data.elem_conc_by_S[-1]
    conc_corrI = data.elem_conc_by_Icorr[-1]

    if data.dTheta == 1.0:
        color = "blue"
    else:
        color = "red"
    row = targets.index(calc.split("_")[-1].split(".")[0])
    column = incident_atoms.index(data.incident_atom)
    axs[row , column].plot(data.scattering_angle, conc_I, "x", color=color)
    axs[row , column].plot(data.scattering_angle, conc_S, "o", color=color)
    axs[row , column].plot(data.scattering_angle, conc_corrI, "*", color=color)
   

# calculate average and std
results = {}
for incident_atom in incident_atoms:
    for target in targets:
        conc_I_all = []
        conc_S_all = []
        conc_corrI_all = []

        for data in datas:
            if incident_atom in data.spectrum_path and target in data.spectrum_path:
                conc_I_all.append(data.elem_conc_by_I[-1])
                if 10<data.elem_conc_by_S[-1]<90: 
                    #due to incorrect automated analysis in case of large angles
                    conc_S_all.append(data.elem_conc_by_S[-1])
                conc_corrI_all.append(data.elem_conc_by_Icorr[-1])
                #if "Au70Pd30" in target and "Kr" in incident_atom and data.dTheta == 1.0:
                #    print(f"{data.scattering_angle} {data.elem_conc_by_S[-1]}")
                
        avg_conc_I = np.mean(conc_I_all) if conc_I_all else 0
        std_conc_I = np.std(conc_I_all) if conc_I_all else 0

        avg_conc_S = np.mean(conc_S_all) if conc_S_all else 0
        std_conc_S = np.std(conc_S_all) if conc_S_all else 0

        avg_conc_corrI = np.mean(conc_corrI_all) if conc_corrI_all else 0
        std_conc_corrI = np.std(conc_corrI_all) if conc_corrI_all else 0

        results[(incident_atom, target)] = {
            'avg_conc_I': avg_conc_I,
            'std_conc_I': std_conc_I,
            'avg_conc_S': avg_conc_S,
            'std_conc_S': std_conc_S,
            'avg_conc_corrI': avg_conc_corrI,
            'std_conc_corrI': std_conc_corrI
        }

"""
for key, value in results.items():
    incident_atom, target = key
    print(f"Incident Atom: {incident_atom}, Target: {target}")
    print(f"  Average conc_I: {value['avg_conc_I']}, Deviation: {value['std_conc_I']}")
    print(f"  Average conc_S: {value['avg_conc_S']}, Deviation: {value['std_conc_S']}")
    print(f"  Average conc_corrI: {value['avg_conc_corrI']}, Deviation: {value['std_conc_corrI']}")
"""

# configure plots
for row in range(0,3):
    for column in range(0,3):
        avg_conc_I = results[(incident_atoms[column], targets[row])]['avg_conc_I']
        std_conc_I = results[(incident_atoms[column], targets[row])]['std_conc_I']
        box = f'C_I={avg_conc_I:.1f}±{std_conc_I:.1f} %\n'
        avg_conc_S = results[(incident_atoms[column], targets[row])]['avg_conc_S']
        std_conc_S = results[(incident_atoms[column], targets[row])]['std_conc_S']
        box += f'C_S={avg_conc_S:.1f}±{std_conc_S:.1f} %\n'
        avg_conc_corrI = results[(incident_atoms[column], targets[row])]['avg_conc_corrI']
        std_conc_corrI = results[(incident_atoms[column], targets[row])]['std_conc_corrI']
        box += f'C_Icorr={avg_conc_corrI:.1f}±{std_conc_corrI:.1f} %'
        
        legend_y_position = 8 if row == 0 or row == 1 else 17
        axs[row, column].text(23, int(targets[row][2:4])-legend_y_position, box, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        axs[row, column].set_xlim(20, 161)
        if row==2 and column == 1:
            axs[row, column].set_xlabel("Scattering angle, deg",  fontsize=14)
        if row!=2:
            axs[row, column].set_xticklabels([])
        if column==0:
            axs[row, column].set_ylabel(targets[row]+'\nAu, conc, %', fontsize=14)
        if row==0:
            axs[row, column].set_title(incident_atoms[column], fontsize=16)
        axs[row, column].set_xticks(np.arange(20, 161, 20))
        axs[row, column].minorticks_on()
        if column!=0:
            axs[row , column].set_yticklabels([])
        axs[row, column].axhline(y=int(targets[row][2:4]), 
                     color='black', linestyle=':', alpha=0.7, linewidth=2)
    
for i in range (0,3):
    axs[0, i].set_ylim(20, 50)
for i in range (0,3):
    axs[1, i].set_ylim(40, 70)
for i in range (0,3):
    axs[2, i].set_ylim(50, 90)

#for i in range (0,3):
 #   axs[i, 0].set_ylabel(targets[i])
    
fig.suptitle("Au concenctration estimations based on peak Intensities (crosses), Areas (circles) and corrected Intensities (stars)", fontsize=20)
axs[0, 0].set_title('Ne', fontsize=16)
axs[0, 1].set_title('Ar', fontsize=16)
axs[0, 2].set_title('Kr', fontsize=16)

plt.show()
