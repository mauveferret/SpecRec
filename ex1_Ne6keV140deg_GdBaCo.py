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

# add some noise to the convoluted sim spectrum
SCD.doBroadeningConvNoise = True
# adding gauss noise where noise_power is a gauss sigma
SCD.noise_power = 0.02

# positions of elastic peaks in eV
E_peak_Co = 1680
# energy where only background is seen, that can be subtracted from Gd and Ba peaks
E_background = 3602
E_peak_Gd = 3766
E_peak_Ba = 3556

# differential cross-sections for elastic scattering peaks in cm2/sr
dSigmadOmega_Co = 0.0026
dSigmadOmega_Ba = 0.00678
dSigmadOmega_Gd = 0.00772

#####################################    CHOOSE INPUT FILES    ######################################

# choose data file
spectra_path = os.getcwd()+os.sep+"raw_data"+os.sep+"ex1_sim_Ne6kev140deg_GdBaCo"
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
    spectrum_en, spectrum_int = SCD.import_data(open(spectra_path+os.sep+datas[f]).read())
    step = SCD.step
    Gd_peak = max(spectrum_int[int((E_peak_Gd-100)/step):int((E_peak_Gd+100)/step)])-spectrum_int[int(E_background/step)]
    Ba_peak = max(spectrum_int[int((E_peak_Ba-100)/step):int((E_peak_Ba+100)/step)])-spectrum_int[int(E_background/step)]
    Co_peak = max(spectrum_int[int((E_peak_Co-100)/step):int((E_peak_Co+100)/step)])
    Co_conc =  (Co_peak/dSigmadOmega_Co)/((Co_peak/dSigmadOmega_Co)+(Gd_peak/dSigmadOmega_Gd)+(Ba_peak/dSigmadOmega_Ba))
    
    data_cnv[0,f+1]=Co_conc
    data_simple_deconv[0,f+1]=Co_conc
    data_numeric_deconv[0,f+1]=Co_conc
    
    for R in range(0,len(SCD.spectrometer_resolutions)):
        
        # do convolution
        conv = SCD.norm(SCD.broadening_kernel_convolution(spectrum_int, spectrum_en, SCD.broadening_kernel_type, 
                                                          SCD.spectrometer_resolutions[R], step))

        Gd_peak = max(conv[int((E_peak_Gd-100)/step):int((E_peak_Gd+100)/step)])-conv[int(E_background/step)]
        Ba_peak = max(conv[int((E_peak_Ba-100)/step):int((E_peak_Ba+100)/step)])-conv[int(E_background/step)]
        Co_peak = max(conv[int((E_peak_Co-100)/step):int((E_peak_Co+100)/step)])
        conv_Co_conc =  (Co_peak/dSigmadOmega_Co)/((Co_peak/dSigmadOmega_Co)+(Gd_peak/dSigmadOmega_Gd)+(Ba_peak/dSigmadOmega_Ba))
        data_cnv[R+1, f+1] = conv_Co_conc
        
        # do simple deconvolution
        simple_deconv = SCD.norm(SCD.simple_deconvolution(conv))
            
        Gd_peak = max(simple_deconv[int((E_peak_Gd-100)/step):int((E_peak_Gd+100)/step)])-simple_deconv[int(E_background/step)]
        Ba_peak = max(simple_deconv[int((E_peak_Ba-100)/step):int((E_peak_Ba+100)/step)])-simple_deconv[int(E_background/step)]
        Co_peak = max(simple_deconv[int((E_peak_Co-100)/step):int((E_peak_Co+100)/step)])
        deconv_Co_conc =  (Co_peak/dSigmadOmega_Co)/((Co_peak/dSigmadOmega_Co)+(Gd_peak/dSigmadOmega_Gd)+(Ba_peak/dSigmadOmega_Ba))
        data_simple_deconv[R+1, f+1] = deconv_Co_conc
        
        #Do more direct deconvolution by solving Fredholm equation with broadening kernel 
        numerical_deconv  = SCD.norm(SCD.twomey_deconvolution(conv, spectrum_en, SCD.broadening_kernel_type, SCD.spectrometer_resolutions[R]))

        Gd_peak = max(numerical_deconv[int((E_peak_Gd-100)/step):int((E_peak_Gd+100)/step)])-numerical_deconv[int(E_background/step)]
        Ba_peak = max(numerical_deconv[int((E_peak_Ba-100)/step):int((E_peak_Ba+100)/step)])-numerical_deconv[int(E_background/step)]
        Co_peak = max(numerical_deconv[int((E_peak_Co-100)/step):int((E_peak_Co+100)/step)])
        numeric_deconv_Co_conc =  (Co_peak/dSigmadOmega_Co)/((Co_peak/dSigmadOmega_Co)+(Gd_peak/dSigmadOmega_Gd)+(Ba_peak/dSigmadOmega_Ba))
        data_numeric_deconv[R+1, f+1] = numeric_deconv_Co_conc

#save data to output and create plots
SCD.save_conc_tables(datas, data_cnv, data_simple_deconv, data_numeric_deconv)
SCD.create_conc_plots(datas, data_cnv, data_simple_deconv, data_numeric_deconv) 