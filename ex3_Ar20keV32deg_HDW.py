import os
# changing working directoru to the location of py file
os.chdir(os.path.dirname(os.path.realpath(__file__))) 
import numpy as np
import spectraConvDeconvLib as SCD

"""
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

If you have questions regarding this program, please contact NEEfimov@mephi.ru
"""

##################################### PRESET SOME CALC PARAMS  #####################################

# smooth of input spectra with a Savitzky-Golay filter 
SCD.doInputSmooth = False
# the width of the filter window for polynomial fitting, in eV
SCD.filter_window_length = 100

#add some noise to the convoluted sim spectrum
SCD.doBroadeningConvNoise = False
#adding gauss noise where noise_power is a gauss sigma
SCD.noise_power = 0.02

# positions of elastic peaks: Scattering of Ar in W and recoil of W in eV
E_peak_H = 1371
E_peak_W = 18718

# energy where only background is seen, that can be subtracted from H peak
E_background_of_H = 1500

# differential cross-sections for elastic scattering peaks
dSigmadOmega_H = 0.0206
dSigmadOmega_W = 0.17

#####################################    CHOOSE INPUT FILES    ######################################

# choose data files
spectra_path = os.getcwd()+os.sep+"raw_data"+os.sep+"ex3_sim_Ar20keV32deg_HDW"
#do not forget to change working dir! 

SCD.calc_name = str(spectra_path.split(os.sep)[-1])
datas = os.listdir(spectra_path)

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
    H_real =  int(datas[f].split("D")[0][1:])
    D_real = 100 - H_real
    
    spectrum_en, spectrum_int = SCD.import_data(open(spectra_path+os.sep+datas[f]).read())
    step = SCD.step
    W_peak = sum(spectrum_int[int((E_peak_W-200)/step):int((E_peak_W+200)/step)])
    without_background = spectrum_int - spectrum_int[int(E_background_of_H/step)]
    H_peak = sum(without_background[int((E_peak_H-100)/step):int((E_peak_H+100)/step)])
    real_H_conc =  H_peak*(dSigmadOmega_W/dSigmadOmega_H)/(H_peak*(dSigmadOmega_W/dSigmadOmega_H)+W_peak)
    
    data_cnv[0,f+1]=real_H_conc
    data_simple_deconv[0,f+1]=real_H_conc
    data_numeric_deconv[0,f+1]=real_H_conc
    
    for R in range(0,len(SCD.spectrometer_resolutions)):
        
        # do convolution
        conv = SCD.norm(SCD.broadening_kernel_convolution(spectrum_int, spectrum_en, SCD.broadening_kernel_type, 
                                                          SCD.spectrometer_resolutions[R], step))

        W_peak = sum(conv[int((E_peak_W-200)/step):int((E_peak_W+200)/step)])  
        without_background = conv - conv[int(E_background_of_H/step)]
        H_peak = sum(without_background[int((E_peak_H-100)/step):int((E_peak_H+100)/step)])
        conv_H_conc =  H_peak*(dSigmadOmega_W/dSigmadOmega_H)/(H_peak*(dSigmadOmega_W/dSigmadOmega_H)+W_peak)
        data_cnv[R+1, f+1] = conv_H_conc
        
        # do simple deconvolution
        simple_deconv = SCD.norm(SCD.simple_deconvolution(conv))
         
        W_peak = sum(simple_deconv[int((E_peak_W-200)/step):int((E_peak_W+200)/step)])
        without_background = simple_deconv - simple_deconv[int(E_background_of_H/step)]
        H_peak = sum(without_background[int((E_peak_H-100)/step):int((E_peak_H+100)/step)])
        deconv_H_conc =  H_peak*(dSigmadOmega_W/dSigmadOmega_H)/(H_peak*(dSigmadOmega_W/dSigmadOmega_H)+W_peak)
        data_simple_deconv[R+1, f+1] = deconv_H_conc
        
        
        #Do more direct deconvolution by solving Fredholm equation with broadening kernel 
        numerical_deconv  = SCD.norm(SCD.twomey_deconvolution(conv, spectrum_en, SCD.broadening_kernel_type, 
                                                              SCD.spectrometer_resolutions[R]))

        W_peak = sum(numerical_deconv[int((E_peak_W-200)/step):int((E_peak_W+200)/step)])  
        without_background = numerical_deconv - numerical_deconv[int(E_background_of_H/step)]
        H_peak = sum(without_background[int((E_peak_H-100)/step):int((E_peak_H+100)/step)])
        numeric_deconv_H_conc =  H_peak*(dSigmadOmega_W/dSigmadOmega_H)/(H_peak*(dSigmadOmega_W/dSigmadOmega_H)+W_peak)
        data_numeric_deconv[R+1, f+1] = numeric_deconv_H_conc

#save data to output and create plots
SCD.save_conc_tables(datas, data_cnv, data_simple_deconv, data_numeric_deconv)
SCD.create_conc_plots(datas, data_cnv, data_simple_deconv, data_numeric_deconv, conc_element_name="H", y_max=101, error_max=60) 
