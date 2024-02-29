"""
This scripts generates figure with estimated relative surface atomic concentrations
of  Co in GdBaCo samples of different composition in dependence on the spectrometer 
relative energy resolution. Data is provided for convoluted and deconvoluted signals 
of simulated spectra of Ne with an energy of 6 keV scattered at an angle of 140° on
the GdBaCo samples. 
The determination of the relative surface concentration was provided by subtraction 
the background from the peaks of elastic scattering and subsequent normalization of 
its intensities by the corresponding differential scattering cross-sections. Points 
at R=0 corresponds to estimations based on the original spectra (A).
Absolute errors due to distortion and reconstruction procedures in the estimation 
of the Co atomic concentrations  in dependence on the spectrometer resolution (B).  


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
SCD.filter_window_length = 50

# add some noise to the convoluted sim spectrum
SCD.doBroadeningConvNoise = False

# type of kernel for broadening kernel: gauss, triangle or rectangle
SCD.broadening_kernel_type = "gauss"

# adding gauss noise where noise_power is a gauss sigma
SCD.noise_power = 0.05

# positions of elastic peaks in eV
E_peak_Co = 1693
E_peak_Gd = 3803
E_peak_Ba = 3556

# energy where only background is seen, that can be subtracted from elastic peaks
E_background =2770

#  elemental sensitivity in a form of difference of squares of impact parameters 
# for a specific registration solid angle divided by a pass energy, Å^2/eV
crossSection_Co = 0.00023/54
crossSection_Ba = 0.00069/47
crossSection_Gd = 0.00061/44

#####################################    CHOOSE INPUT FILES    ######################################

# choose data file
spectra_path = os.getcwd()+os.sep+"raw_data"+os.sep+"ex1_sim_Ne6kev140deg_GdBaCo"

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
    spectrum_en, spectrum_int = SCD.import_data(spectra_path+os.sep+datas[f])
    step = SCD.step
    Gd_peak = max(spectrum_int[int((E_peak_Gd-50)/step):int((E_peak_Gd+50)/step)])-spectrum_int[int(E_background/step)]
    Ba_peak = max(spectrum_int[int((E_peak_Ba-50)/step):int((E_peak_Ba+50)/step)])-spectrum_int[int(E_background/step)]
    Co_peak = max(spectrum_int[int((E_peak_Co-50)/step):int((E_peak_Co+50)/step)])-spectrum_int[int(E_background/step)]
    Co_conc =  (Co_peak/crossSection_Co)/((Co_peak/crossSection_Co)+(Gd_peak/crossSection_Gd)+(Ba_peak/crossSection_Ba))

    data_cnv[0,f+1]=Co_conc
    data_simple_deconv[0,f+1]=Co_conc
    data_numeric_deconv[0,f+1]=Co_conc
    
    for R in range(0,len(SCD.spectrometer_resolutions)):
        
        # do broadening convolution
        conv = SCD.broadening_kernel_convolution(spectrum_int, spectrum_en, SCD.broadening_kernel_type, 
                                                          SCD.spectrometer_resolutions[R])

        Gd_peak = max(conv[int((E_peak_Gd-50)/step):int((E_peak_Gd+50)/step)])-conv[int(E_background/step)]
        Ba_peak = max(conv[int((E_peak_Ba-50)/step):int((E_peak_Ba+50)/step)])-conv[int(E_background/step)]
        Co_peak = max(conv[int((E_peak_Co-50)/step):int((E_peak_Co+50)/step)])-conv[int(E_background/step)]
        conv_Co_conc =  (Co_peak/crossSection_Co)/((Co_peak/crossSection_Co)+(Gd_peak/crossSection_Gd)+(Ba_peak/crossSection_Ba))
        data_cnv[R+1, f+1] = conv_Co_conc
        
        # do simple deconvolution
        simple_deconv = SCD.simple_deconvolution(conv)   
        Gd_peak = max(simple_deconv[int((E_peak_Gd-50)/step):int((E_peak_Gd+50)/step)])-simple_deconv[int(E_background/step)]
        Ba_peak = max(simple_deconv[int((E_peak_Ba-50)/step):int((E_peak_Ba+50)/step)])-simple_deconv[int(E_background/step)]
        Co_peak = max(simple_deconv[int((E_peak_Co-50)/step):int((E_peak_Co+50)/step)])-simple_deconv[int(E_background/step)]
        deconv_Co_conc =  (Co_peak/crossSection_Co)/((Co_peak/crossSection_Co)+(Gd_peak/crossSection_Gd)+(Ba_peak/crossSection_Ba))
        data_simple_deconv[R+1, f+1] = deconv_Co_conc
        
        #Do more direct deconvolution by solving Fredholm equation with broadening kernel 
        numerical_deconv  = SCD.twomey_deconvolution(conv, spectrum_en, SCD.broadening_kernel_type, SCD.spectrometer_resolutions[R])
        Gd_peak = max(numerical_deconv[int((E_peak_Gd-50)/step):int((E_peak_Gd+50)/step)])-numerical_deconv[int(E_background/step)]
        Ba_peak = max(numerical_deconv[int((E_peak_Ba-50)/step):int((E_peak_Ba+50)/step)])-numerical_deconv[int(E_background/step)]
        Co_peak = max(numerical_deconv[int((E_peak_Co-50)/step):int((E_peak_Co+50)/step)])-numerical_deconv[int(E_background/step)]
        numeric_deconv_Co_conc =  (Co_peak/crossSection_Co)/((Co_peak/crossSection_Co)+(Gd_peak/crossSection_Gd)+(Ba_peak/crossSection_Ba))
        data_numeric_deconv[R+1, f+1] = numeric_deconv_Co_conc

#save data to output and create plots
SCD.save_conc_tables(datas, data_cnv, data_simple_deconv, data_numeric_deconv)
SCD.create_conc_plots(datas, data_cnv, data_simple_deconv, data_numeric_deconv, y_max=92, error_max=20) 