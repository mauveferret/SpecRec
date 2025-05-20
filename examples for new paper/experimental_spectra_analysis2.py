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
import matplotlib.pyplot as plt2
# changing working directoru to the SpecRec dir
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(parent_dir) 
BASE_LIB_PATH = "tools"
sys.path.insert(1, os.getcwd()+os.sep+BASE_LIB_PATH)
import LEIS_tools as leis
#import spectraConvDeconv_tools as SCD

# Presets
spectrum_path0 = os.getcwd()+os.sep+"raw_data"+os.sep+"exp_AuPd"
leis.Emin = 5000 # eV
leis.Emax = 15000 # eV
dE = 2 # eV
# SCD.step = dE # eV
filter_window = 100 # eV
# R - relative energy resolution of spectrometer
R = 0.01
plt.figure(figsize=(12, 8))

# Load reference spectra
exp_spectra = os.listdir(spectrum_path0)
for spectrum in exp_spectra:
    if "ref_Ne_Au" in spectrum:
        ref_Ne_Au = leis.spectrum(spectrum_path0+os.sep+spectrum, filter_window, step=dE)
    if "ref_Ar_Au" in spectrum:
        ref_Ar_Au = leis.spectrum(spectrum_path0+os.sep+spectrum, filter_window, step=dE)
        
# Load experimental spectra and calculate the concentration of Au and Pd
i = 0
i_ar = 0
i_ne = 0
for spectrum in exp_spectra:
    if not "ref" in spectrum:
        data = leis.spectrum(spectrum_path0+os.sep+spectrum, filter_window, step=dE)
        if "Ne" in data.incident_atom:
            Pd_signal = leis.norm(data.spectrum_int) - np.interp(data.spectrum_en, ref_Ne_Au.spectrum_en, ref_Ne_Au.spectrum_int)
        elif "Ar" in data.incident_atom:
            Pd_signal = leis.norm(data.spectrum_int) - np.interp(data.spectrum_en, ref_Ar_Au.spectrum_en, ref_Ar_Au.spectrum_int)
        else:
            Pd_signal = leis.norm(data.spectrum_int)
            print(f"No reference was found for the {data.incident_atom} incident atom")
       # Calculate the concentration of Au and Pd based on the SemiRef approach and the sensitivity factors

    #plt2.plot(data.spectrum_en, leis.norm(data.spectrum_int), label="raw")
    #plt2.plot(data.spectrum_en, Pd_signal, label="Pd")
    #if "Ne" in data.incident_atom:
    #    plt2.plot(ref_Ne_Au.spectrum_en, ref_Ne_Au.spectrum_int, label="Au")
   # else:
    #    plt2.plot(ref_Ar_Au.spectrum_en, ref_Ar_Au.spectrum_int, label="Au")


    int_Pd = leis.peak(Pd_signal)/leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Pd")
    int_Au = 1/leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Au")
    
    conc_Au_semiRef_cross = int_Au/(int_Au+int_Pd)*100
    
    # Calculate the concentration of Au and Pd based on the Young's fitting model 
    young_fitting = leis.fitted_spectrum(data, "Pd", "Au")
    conc_Au_fitting = young_fitting.get_concentration()
    
    plt2.plot(data.spectrum_en, young_fitting.get_fitted_spectrum(), label="Young")
    box  = f"Au_conc_by_Young_fit = {conc_Au_fitting:.2f} at. % \nAu_conc_by_SemiRef = {conc_Au_semiRef_cross:.2f} at. %"   
    plt2.text(11000, 0.7, box, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    plt.ylim(0, 1)
    plt.xlim(10000,15000)
    plt.title(f"Experimental energy spectrum for {spectrum}")
    plt2.show(block=True)
    plt2.pause(0.01)
    print(f"{data.calc_name[0:16]} {data.incident_atom} {conc_Au_semiRef_cross:.2f} % {conc_Au_fitting:.2f} %")
    if "Ne" in data.incident_atom:
        i_ne+=1        
        #plt.plot(i_ne, conc_Au_semiRef_cross, "x", color="red" if data.incident_atom == "Ne" else "green")
        #plt.plot(i_ne, conc_Au_fitting, "o", color="red" if data.incident_atom == "Ne" else "green")
    else:
        i_ar+=1
        #plt.plot(i_ar, conc_Au_semiRef_cross, "x", color="red" if data.incident_atom == "Ne" else "green")
        #plt.plot(i_ar, conc_Au_fitting, "o", color="red" if data.incident_atom == "Ne" else "green")
    # Store concentrations for statistics
    if i == 0:
        Ne_conc = []
        Ar_conc = []
    
    if data.incident_atom == "Ne":
        Ne_conc.append(conc_Au_semiRef_cross)
    else:
        Ar_conc.append(conc_Au_semiRef_cross)

    i+=1

    # After last spectrum, print statistics
    if i == len([s for s in exp_spectra if "ref" not in s]):
        if Ne_conc:
            print(f"\nNe incident atoms:")
            print(f"Average Au concentration: {np.mean(Ne_conc):.2f}%")
            print(f"Standard deviation: {np.std(Ne_conc):.2f}%")
        if Ar_conc:
            print(f"\nAr incident atoms:")
            print(f"Average Au concentration: {np.mean(Ar_conc):.2f}%")
            print(f"Standard deviation: {np.std(Ar_conc):.2f}%")

plt.axhline(y=50, color='black', linestyle=':', alpha=0.7, linewidth=2)
plt.ylim(40, 70)
plt.xlabel('номер спектра')
plt.ylabel('концентрация Au, %')
plt.title('Concentration of Au in the Au50Pd50 samples for experimental LEIS spectra. Ne - RED, Ar - GREEN')
plt.grid(True)
plt.minorticks_on()
plt.legend([ "Полуэталонный метод", "Полуэмппирическая аппроцкцсимация"])
plt.show()
