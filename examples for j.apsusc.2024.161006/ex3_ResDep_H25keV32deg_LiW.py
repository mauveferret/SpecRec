
"""
This script is intended for generating dependency of Li film thickness 
and error in its determination on the relative energy resolution of the spectrometer

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

If you have questions regarding this program, please contact NEEfimov@mephi.ru
"""

import os, sys
# changing working directoru to the SpecRec dir
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(parent_dir) 
# Add parent directory to sys.path
sys.path.append(parent_dir)
import numpy as np

BASE_LIB_PATH = "tools"
sys.path.insert(1, os.getcwd()+os.sep+BASE_LIB_PATH)
import LEIS_tools as leis
import spectraConvDeconv_tools as SCD

##################################### PRESET SOME CALC PARAMS  #####################################

# smooth of input spectra with a Savitzky-Golay filter 
SCD.doInputSmooth = True
# the width of the filter window for polynomial fitting, in eV
SCD.filter_window_length = 300

# type of kernel for broadening kernel: gauss, triangle or rectangle
SCD.broadening_kernel_type = "gauss"

# add some noise to the convoluted sim spectrum
SCD.doBroadeningConvNoise = False
# adding gauss noise where noise_power is a gauss sigma
SCD.noise_power = 0.02
SCD.spectrometer_resolutions = ( 0.005, 0.008, 0.01, 0.012,0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05)

concs_to_calculate = ("Li20","Li30","Li40","Li50", "Li60", "Li80")

#####################################    CHOOSE INPUT FILES    ######################################

# choose data files
spectra_path = os.getcwd()+os.sep+"raw_data"+os.sep+"ex3_sim_H25keV32deg_LiW"

SCD.calc_name = str(spectra_path.split(os.sep)[-1])
files = os.listdir(spectra_path)
datas = []
legend = []
for file in files:
    for conc in concs_to_calculate:
        if conc in file:
            datas.append(file)
            legend.append(conc.replace("Li","Li=")+" nm")
            break;
        
# arrays for output data
data_cnv = np.zeros((len(SCD.spectrometer_resolutions)+1,len(datas)+1))
data_simple_deconv = np.zeros((len(SCD.spectrometer_resolutions)+1,len(datas)+1))
data_numeric_deconv = np.zeros((len(SCD.spectrometer_resolutions)+1,len(datas)+1))


# calibration based on the polynomial fitting of peaks position in  H->WLi simulation
def thick_calib(peak_pos):
    #peak_pos in eV, thickness in nm
    return 90.17442-0.01065*peak_pos+9.99068E-7*peak_pos**2-4.35214E-11*peak_pos**3+5.36148E-16*peak_pos**4

#write X axis in data arrays
for R in range(0,len(SCD.spectrometer_resolutions)):
    data_cnv[R+1,0] = SCD.spectrometer_resolutions[R]
    data_simple_deconv[R+1,0] = SCD.spectrometer_resolutions[R]
    data_numeric_deconv[R+1,0] = SCD.spectrometer_resolutions[R]

for f in range(0, len(datas)):    
    spectrum_en, spectrum_int = SCD.import_data(spectra_path+os.sep+datas[f])
    step = SCD.step 

    W_peak_max = SCD.peak(spectrum_int)      
    W_peak_pos = 500 #approximate
    for E in range (0, len(spectrum_int)):            
        if (abs(spectrum_int[E]-W_peak_max))<0.001:
            W_peak_pos = E
    W_peak_pos *=step
    data_cnv[0,f+1]=thick_calib(W_peak_pos)
    data_simple_deconv[0,f+1]=thick_calib(W_peak_pos)
    data_numeric_deconv[0,f+1]=thick_calib(W_peak_pos)
    
    for R in range(0,len(SCD.spectrometer_resolutions)):
    
        # do convolution
        conv = SCD.broadening_kernel_convolution(spectrum_int, spectrum_en, SCD.broadening_kernel_type, 
                                                 SCD.spectrometer_resolutions[R])
        W_peak_max =  SCD.peak(conv)      
        W_peak_pos = 500 #approximate
        for E in range (0, len(spectrum_int)):  
            if (abs(conv[E]-W_peak_max))<0.001:
                W_peak_pos = E
        W_peak_pos *=step
        data_cnv[R+1, f+1] = thick_calib(W_peak_pos)

        # do simple deconvolution
        simple_deconv = SCD.simple_deconvolution(conv)
        W_peak_max =  SCD.peak(simple_deconv) 
        W_peak_pos = 500 #approximate
        for E in range (0, len(spectrum_int)):  
            if (abs(simple_deconv[E]-W_peak_max))<0.001:
                W_peak_pos = E
        W_peak_pos *=step
        data_simple_deconv[R+1, f+1] = thick_calib(W_peak_pos)
        
        #Do more direct deconvolution by solving Fredholm equation with broadening kernel 
        numerical_deconv  = SCD.twomey_deconvolution(conv, spectrum_en, SCD.broadening_kernel_type, SCD.spectrometer_resolutions[R])
        W_peak_max =  SCD.peak(numerical_deconv) 
        W_peak_pos = 500 #approximate
        for E in range (0, len(spectrum_int)):  
            if (abs(numerical_deconv[E]-W_peak_max))<0.001:
                W_peak_pos = E
        W_peak_pos *=step
        data_numeric_deconv[R+1, f+1] = thick_calib(W_peak_pos)
        
#save data to output and create plots
SCD.save_conc_tables(datas, data_cnv, data_simple_deconv, data_numeric_deconv,type = "thickness")
SCD.create_conc_plots(legend, data_cnv, data_simple_deconv, data_numeric_deconv, type="thickness",
                      conc_element_name="Li", y_max=85, y1_step=10, error_max=6) 
