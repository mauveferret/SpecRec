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

spectrum_path0 = os.getcwd()+os.sep+"raw_data"+os.sep+"exp_AuPd"
dE=SCD.step

exp_spectra = os.listdir(spectrum_path0)
for spectrum in exp_spectra:
    if "ref_Ne_Au" in spectrum:
        ref_Ne_Au = leis.spectrum(spectrum_path0+os.sep+spectrum, 20)
    if "ref_Ar_Au" in spectrum:
        ref_Ar_Au = leis.spectrum(spectrum_path0+os.sep+spectrum, 20)

for spectrum in exp_spectra:
    if not "ref" in spectrum:
        data = leis.spectrum(spectrum_path0+os.sep+spectrum, 20)
        if "Ne" in data.incident_atom:
            
            Pd_signal = data.spectrum_int - np.interp(data.spectrum_en, ref_Ne_Au.spectrum_en, ref_Ne_Au.spectrum_int)
        elif "Ar" in data.incident_atom:
            Pd_signal = data.spectrum_int - np.interp(data.spectrum_en, ref_Ar_Au.spectrum_en, ref_Ar_Au.spectrum_int)
        else:
            Pd_signal = data.spectrum_int
            print("No reference was found for the "+data.incident_atom+" incident atom")
    int_Pd = leis.peak(Pd_signal)*leis.get_sensitivity_factor(data.E0, data.incident_atom, "Pd", data.theta,data.dTheta)
    int_Au = 1*leis.get_sensitivity_factor(data.E0, data.incident_atom, "Au", data.theta,data.dTheta)
    print(data.calc_name[0:15]+" "+data.incident_atom+" "+" conc:"+str(int_Au/(int_Au+int_Pd)*100)[0:4]+"%")
    

#leis.plot_spectrum(ref_Ar_Au)