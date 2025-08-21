import os, sys
import numpy as np
import matplotlib.pyplot as plt
#change path to working directory
BASE_LIB_PATH = "tools"
sys.path.insert(1, os.getcwd()+os.sep+BASE_LIB_PATH)
import LEIS_tools as leis
# set the potential to ZBL for differential cross section calculation. 
# TFM and KRC are also available
leis.set_potential("ZBL")
incident_element = "Ne"
E0 = 15000 # eV
dE = 150 # eV

target_elements = ["Au", "Pd"]
colors = ['red', 'blue', 'green', 'orange', 'black', 'purple']
angles = range(5, 175, 2)

fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(8, 16))
plt.subplots_adjust(hspace=0.1, wspace=0.1)

#ax1.plot([], [], ' ', label="cross sections (ZBL):")
#ax2.plot([], [], ' ', label="detection solid angles:")

i=0

for target_element in target_elements:
    cross_sections = np.zeros(len(angles))
    Peak_pos = np.zeros(len(angles))
    for angle in angles:
        cross_sections[angles.index(angle)] = leis.get_cross_section(incident_element, E0,angle, target_element)
        Peak_pos[angles.index(angle)] = leis.get_energy_by_angle(E0, leis.get_mass_by_element(target_element)/leis.get_mass_by_element(incident_element), angle)

    ax1.plot(angles, cross_sections, linestyle = "-", linewidth=3, color = colors[i], label=f'{incident_element} {E0/1000:.1f} keV → {target_element}')
    
    ax2.plot(angles, Peak_pos/1000, linestyle = "-", linewidth=3, color=colors[i], label=f'Elastic peak position for {target_element}')
    i+=1


plt.xlim(5, 175)
ax1.set_yscale('log')
ax1.set_ylim(0.001, 10)

ax1.tick_params(axis='y', labelsize=16)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

ax2.set_ylim(6, 15)
ax2.set_yscale('linear')
ax1.set_ylabel(r'dσ/dΩ, Å$^2$/sr', fontsize=18)
ax2.set_ylabel('energy, keV', fontsize=18)
ax2.set_xlabel('scattering angle, deg.', fontsize=18)
plt.minorticks_on()
# Add legends for both axes
#lines, labels = ax1.get_legend_handles_labels()
#lines2, labels2 = ax2.get_legend_handles_labels()
# Combine legends
#lines_combined = lines + lines2
#labels_combined = labels + labels2
ax1.legend(loc="lower left", frameon=False, fontsize=18)
ax2.legend(loc="lower left", frameon=False, fontsize=18)
# Remove the top spine
ax1.spines['top'].set_visible(False)

# Optionally, remove the right spine as well for a cleaner look
ax1.spines['right'].set_visible(False)

# Remove the top spine
ax2.spines['top'].set_visible(False)

# Optionally, remove the right spine as well for a cleaner look
ax2.spines['right'].set_visible(False)
plt.show() 