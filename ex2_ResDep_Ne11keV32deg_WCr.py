
"""
This scripts generates figure with estimated relative surface concentrations 
of Cr in W-Cr samples of different composition in dependence on the spectrometer 
resolution. Data is provided for convoluted and deconvoluted signals of simulated 
spectra of Ne with an energy of 11 keV scattered at an angle of 32° on the W-Cr samples.
Points at R=0 corresponds to estimations based on the raw spectra (A). Absolute errors due 
to distortion and reconstruction procedures in the estimation of the Cr atomic concentrations 
in dependence on the spectrometer resolution (B).

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
import numpy as np
import spectraConvDeconvLib as SCD

##################################### PRESET SOME CALC PARAMS  #####################################

# smooth of input spectra with a Savitzky-Golay filter 
SCD.doInputSmooth = True
# the width of the filter window for polynomial fitting, in eV
SCD.filter_window_length = 100

# type of kernel for broadening kernel: gauss, triangle or rectangle
SCD.broadening_kernel_type = "gauss"

SCD.spectrometer_resolutions = ( 0.0008, 0.001, 0.002, 0.005, 0.008, 0.01, 0.012,0.015, 0.02, 0.025, 0.03, 0.035)


# add some noise to the convoluted sim spectrum
SCD.doBroadeningConvNoise = False
# adding gauss noise where noise_power is a gauss sigma
SCD.noise_power = 0.02

# influence on search of the peaks maxima. True value alloes to minimaze error due to
# peaks shifting
doBroadeningVicinityFroMax = True

# positions of elastic peaks: Scattering of Ne on Cr and W 
E_peak_Cr = 9760
E_peak_W = 10640

energy_width_original = E_peak_W*0.008/2

#  elemental sensitivity in a form of difference of squares of impact parameters 
# for a specific registration solid angle, Å^2
crossSection_Cr = 0.00327
crossSection_W = 0.00688


concs_to_calculate = ( "Cr0", "Cr10","Cr30","Cr50", "Cr70", "Cr90", "Cr100")

#####################################    CHOOSE INPUT FILES    ######################################

# choose data files
spectra_path = os.getcwd()+os.sep+"raw_data"+os.sep+"ex2_sim_Ne11keV32deg_WCr"

SCD.calc_name = str(spectra_path.split(os.sep)[-1])
files = os.listdir(spectra_path)
datas = []
legend = []

for conc in concs_to_calculate:
    for file in files:
        if conc+".dat"  in file:
            datas.append(file)
            legend.append(conc.replace("Cr","Cr=")+"%")
            break;

if doBroadeningVicinityFroMax:
    SCD.logging_options+="_BroadenVicinity_"

# arrays with backgrounds to subtract from E_peak1
background_ref = np.zeros(12000)
background_ref_conv = np.zeros((len(SCD.spectrometer_resolutions), 12000))
background_ref_simple_deconv= np.zeros((len(SCD.spectrometer_resolutions), 12000))
background_ref_numeric_deconv= np.zeros((len(SCD.spectrometer_resolutions), 12000))

# arrays for output data
data_cnv = np.zeros((len(SCD.spectrometer_resolutions)+1,len(datas)+1))
data_simple_deconv = np.zeros((len(SCD.spectrometer_resolutions)+1,len(datas)+1))
data_numeric_deconv = np.zeros((len(SCD.spectrometer_resolutions)+1,len(datas)+1))

#write X axis in data arrays
for R in range(0,len(SCD.spectrometer_resolutions)):
    
    data_cnv[R+1,0] = SCD.spectrometer_resolutions[R]
    data_simple_deconv[R+1,0] = SCD.spectrometer_resolutions[R]
    data_numeric_deconv[R+1,0] = SCD.spectrometer_resolutions[R]

for f in range(0, len(datas)):    
    W_real =  int(datas[f].split("Cr")[0][1:])
    Cr_real = 100 - W_real
    
    spectrum_en, spectrum_int = SCD.import_data(spectra_path+os.sep+datas[f])
    step = SCD.step
    
    if W_real == 100:
        background_ref[0:len(spectrum_int)] = spectrum_int
    W_peak = max(spectrum_int[int((E_peak_W-energy_width_original)/step):int((E_peak_W+energy_width_original)/step)]) 
    Cr_peak_sp = (spectrum_int-background_ref[0:len(spectrum_int)]*W_peak)
    Cr_peak = max(Cr_peak_sp[int((E_peak_Cr-energy_width_original)/step):int((E_peak_Cr+energy_width_original)/step)])
    real_Cr_conc =  Cr_peak*(crossSection_W/crossSection_Cr)/(Cr_peak*(crossSection_W/crossSection_Cr)+W_peak)
    
    data_cnv[0,f+1]=real_Cr_conc
    data_simple_deconv[0,f+1]=real_Cr_conc
    data_numeric_deconv[0,f+1]=real_Cr_conc
    
    for R in range(0,len(SCD.spectrometer_resolutions)):
        
        if doBroadeningVicinityFroMax:
            energy_width=E_peak_W*SCD.spectrometer_resolutions[R]/2
        else:
            energy_width=E_peak_W*0.008/2 
            
        # do broadening convolution
        conv = SCD.broadening_kernel_convolution(spectrum_int, spectrum_en, SCD.broadening_kernel_type, 
                                                          SCD.spectrometer_resolutions[R])     
        if W_real == 100:
            background_ref_conv[R][0:len(conv)] = conv
        W_peak = max(conv[int((E_peak_W-energy_width)/step):int((E_peak_W+energy_width)/step)])   
        Cr_peak_sp = (conv-background_ref_conv[R][0:len(conv)]*W_peak)
        Cr_peak = max(Cr_peak_sp[int((E_peak_Cr-energy_width)/step):int((E_peak_Cr+energy_width)/step)])
        conv_Cr_conc =  Cr_peak*(crossSection_W/crossSection_Cr)/(Cr_peak*(crossSection_W/crossSection_Cr)+W_peak)
        data_cnv[R+1, f+1] = conv_Cr_conc
        
        # do simple deconvolution
        simple_deconv = SCD.simple_deconvolution(conv)
        
        if W_real == 100:
            background_ref_simple_deconv[R][0:len(simple_deconv)] = simple_deconv
        W_peak = max(simple_deconv[int((E_peak_W-energy_width)/step):int((E_peak_W+energy_width)/step)])    
        Cr_peak_sp = (simple_deconv-background_ref_simple_deconv[R][0:len(simple_deconv)]*W_peak)
        Cr_peak = max(Cr_peak_sp[int((E_peak_Cr-energy_width)/step):int((E_peak_Cr+energy_width)/step)])
        deconv_Cr_conc =  Cr_peak*(crossSection_W/crossSection_Cr)/(Cr_peak*(crossSection_W/crossSection_Cr)+W_peak)
        data_simple_deconv[R+1, f+1] = deconv_Cr_conc
        
        #Do more direct deconvolution by solving Fredholm equation with broadening kernel 
        numerical_deconv  = SCD.twomey_deconvolution(conv, spectrum_en, SCD.broadening_kernel_type, SCD.spectrometer_resolutions[R])
        if W_real == 100:
            background_ref_numeric_deconv [R][0:len(numerical_deconv)] = numerical_deconv
        W_peak = max(numerical_deconv[int((E_peak_W-energy_width)/step):int((E_peak_W+energy_width)/step)])  
        Cr_peak_sp = (numerical_deconv-background_ref_numeric_deconv[R][0:len(numerical_deconv)]*W_peak)
        Cr_peak = max(Cr_peak_sp[int((E_peak_Cr-energy_width)/step):int((E_peak_Cr+energy_width)/step)])
        numeric_deconv_Cr_conc =  Cr_peak*(crossSection_W/crossSection_Cr)/(Cr_peak*(crossSection_W/crossSection_Cr)+W_peak)
        data_numeric_deconv[R+1, f+1] = numeric_deconv_Cr_conc

#save data to output and create plots
SCD.save_conc_tables(datas, data_cnv, data_simple_deconv, data_numeric_deconv)
SCD.create_conc_plots(legend, data_cnv, data_simple_deconv, data_numeric_deconv,
                      conc_element_name="Cr", y_max=101, error_max=7) 