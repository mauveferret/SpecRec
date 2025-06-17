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
from scipy.optimize import minimize

# changing working directoru to the SpecRec dir
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(parent_dir) 
BASE_LIB_PATH = "tools"
sys.path.insert(1, os.getcwd()+os.sep+BASE_LIB_PATH)
import LEIS_tools as leis
import spectraConvDeconv_tools as SCD

dE = 2 # eV
leis.step = dE # eV
SCD.step = dE 

##############################            PRESETS          ##########################################################

spectrum_path0 = os.getcwd()+os.sep+"raw_data"+os.sep+"ex2_sim_Ne11keV32deg_WCr"

leis.Emin = 7000 # eV
leis.Emax = 11000 # eV

# smoothing parameter
filter_window = 50 # eV

# R - relative energy resolution of spectrometer
R = 0.01
 
do_spectra_charts = True

####################################################################################################################

#plt.figure(figsize=(12, 8))

# Load reference spectra
sim_spectra = os.listdir(spectrum_path0)

# Load experimental spectra and calculate the concentration of Au and Pd
i = 0
i_ne = 0
for spectrum in sim_spectra:

        
    data = leis.spectrum(spectrum_path0+os.sep+spectrum, filter_window, step=dE)
        
    # Calculate the concentration of Au and Pd based on the Young's fitting model 
    young_fitting = leis.fitted_spectrum(data, "Cr", "W")
    conc_Au_fitting = young_fitting.get_conc_by_inten()



    if do_spectra_charts:
        plt.figure(figsize=(12, 8))
        plt.plot(data.spectrum_en/1000, leis.norm(data.spectrum_int), "k-", label="Экспериментальный спектр Au50Pd50", linewidth=3, alpha=0.9)
        box  = f"Концентрация золота \n Янг = {conc_Au_fitting:.2f} ат. % "   

        plt.text(10.2, 0.5, box, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

        plt.plot(data.spectrum_en/1000, young_fitting.get_fitted_spectrum(), label="Аппроксимация по формуле Йанга")
        plt.plot(data.spectrum_en/1000, young_fitting.get_elastic_part("W"), "--", label="Упругая часть Au по формуле Йанга")
        plt.plot(data.spectrum_en/1000, young_fitting.get_inelastic_part("W"), "--", label="Неупругая часть Au по формуле Йанга")     
        plt.plot(data.spectrum_en/1000, young_fitting.get_elastic_part("Cr"), "--", label="Упругая часть Pd по формуле Йанга")
        plt.plot(data.spectrum_en/1000, young_fitting.get_inelastic_part("Cr"), "--", label="Неупругая часть Pd по формуле Йанга")      
        plt.xlim(6,11)
        plt.ylim(0, 1)
        plt.xlabel('энергия, кэВ', fontsize=16)
        plt.ylabel('интенсивность, норм.', fontsize=16)
        plt.title(f"Экспериментальный спектр {spectrum}", y=1.05) 
        plt.legend(fontsize=11)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.minorticks_on()
        plt.show()
    print (f"{spectrum} {conc_Au_fitting}")
if not do_spectra_charts:
    #plt.plot(-1, 0, "*", color= "green", label ="Аргон 15 кэВ")
    plt.plot(-1, 0, "x", color="red", label ="Ne 15 кэВ \"Полуэталонный\"")
    #plt.plot(-1, 0, "o", color="red", label ="Ne 15 кэВ \"Эталонный\" норм.")
    plt.plot(-1, 0, "*", color="green", label ="Ar 15 кэВ \"Полуэталонный\"")
    plt.plot(-1, 0, "o", color="blue", label ="пик золота / 1E-8 * 100%")
    plt.plot(-1, 0, "<", color="black", label ="пик палладия / 1E-8 * 100%")

    plt.axhline(y=50, color='black', linestyle=':', alpha=0.8, linewidth=3)
    plt.xlim(left=0)
    plt.ylim(0, 120)
    plt.xlabel('номер спектра', fontsize = 15)
    plt.ylabel('концентрация Au, %', fontsize = 15)
    plt.title(f'Concentration of Au in the Au50Pd50 samples for experimental LEIS spectra. \n Sample Temperature is shown in Annotations \n {spectrum_path0}', y=1.02)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.minorticks_on()    
    plt.legend( fontsize = 15)
    plt.show()
