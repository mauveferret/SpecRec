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
SCD.doInputSmooth = True
# the width of the filter window for polynomial fitting, in eV
SCD.filter_window_length = 500

# type of kernel for broadening kernel: gauss, triangle or rectangle
SCD.broadening_kernel_type = "gauss"

# add some noise to the convoluted sim spectrum
SCD.doBroadeningConvNoise = False
# adding gauss noise where noise_power is a gauss sigma
SCD.noise_power = 0.02

#####################################    CHOOSE INPUT FILES    ######################################

# choose data files
spectra_path = os.getcwd()+os.sep+"raw_data"+os.sep+"ex31_sim_H18keV38deg_AuSi"
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

    Au_peak_max =  SCD.peak(spectrum_int)  
    Au_peak_pos = 15400 #approximate
    E1_FWHM = 0
    E2_FWHM = 0
    for E in range (12000, int(spectrum_en[-1])):            
        if (E<Au_peak_pos and abs(spectrum_int[int(E/step)]-Au_peak_max/2)<0.02):
            E1_FWHM = E
        if (abs(spectrum_int[int(E/step)]-Au_peak_max))<0.02:
            Au_peak_pos = E
        if (E>Au_peak_pos and abs(spectrum_int[int(E/step)]-Au_peak_max/2)<0.02):
            E2_FWHM = E
    #print(datas[f]+" "+str(E1_FWHM)+" "+str(Au_peak_pos)+" "+str(E2_FWHM))
    width = (E2_FWHM-E1_FWHM)/1000

    calib_coeff = float(datas[f].split("u")[-1].split("n")[0].replace('\t'," ").replace(",", "."))/width
    width *=calib_coeff 
    data_cnv[0,f+1]=width
    data_simple_deconv[0,f+1]=width
    data_numeric_deconv[0,f+1]=width
    
    for R in range(0,len(SCD.spectrometer_resolutions)):
        
        # do convolution
        conv = SCD.norm(SCD.broadening_kernel_convolution(spectrum_int, spectrum_en, SCD.broadening_kernel_type, 
                                                          SCD.spectrometer_resolutions[R], step))
        Au_peak_max =  SCD.peak(conv)   
        Au_peak_pos = 15400 #approximate
        E1_FWHM = 0
        E2_FWHM = 0
        for E in range (12000, int(spectrum_en[-1])):
            if (E<Au_peak_pos and abs(conv[int(E/step)]-Au_peak_max/2)<0.02):
                E1_FWHM = E
            if (abs(conv[int(E/step)]-Au_peak_max))<0.02:
                Au_peak_pos = E
            if (E>Au_peak_pos and abs(conv[int(E/step)]-Au_peak_max/2)<0.02):
                E2_FWHM = E
        width = (E2_FWHM-E1_FWHM)/1000
        width *=calib_coeff 
        data_cnv[R+1, f+1] = width
        
        # do simple deconvolution
        simple_deconv = SCD.norm(SCD.simple_deconvolution(conv))
        Au_peak_max =  SCD.peak(simple_deconv)    
        Au_peak_pos = 15400 #approximate
        E1_FWHM = 0
        E2_FWHM = 0
        for E in range (12000, int(spectrum_en[-1])):
            if (E<Au_peak_pos and abs(simple_deconv[int(E/step)]-Au_peak_max/2)<0.02):
                E1_FWHM = E
            if (abs(simple_deconv[int(E/step)]-Au_peak_max))<0.02:
                Au_peak_pos = E
            if (E>Au_peak_pos and abs(simple_deconv[int(E/step)]-Au_peak_max/2)<0.02):
                E2_FWHM = E
        width = (E2_FWHM-E1_FWHM)/1000
        width *=calib_coeff 
        data_simple_deconv[R+1, f+1] = width
             
        #Do more direct deconvolution by solving Fredholm equation with broadening kernel 
        numerical_deconv  = SCD.norm(SCD.twomey_deconvolution(conv, spectrum_en, SCD.broadening_kernel_type, SCD.spectrometer_resolutions[R]))

        Au_peak_max =  SCD.peak(numerical_deconv)   
        Au_peak_pos = 15400 #approximate
        E1_FWHM = 0
        E2_FWHM = 0
        for E in range (12000, int(spectrum_en[-1])):
            if (E<Au_peak_pos and abs(numerical_deconv[int(E/step)]-Au_peak_max/2)<0.02):
                E1_FWHM = E
            if (abs(numerical_deconv[int(E/step)]-Au_peak_max))<0.02:
                Au_peak_pos = E
            if (E>Au_peak_pos and abs(numerical_deconv[int(E/step)]-Au_peak_max/2)<0.02):
                E2_FWHM = E
        width = (E2_FWHM-E1_FWHM)/1000
        width *=calib_coeff 
        data_numeric_deconv[R+1, f+1] = width
        
#save data to output and create plots
SCD.save_conc_tables(datas, data_cnv, data_simple_deconv, data_numeric_deconv)
SCD.create_conc_plots(datas, data_cnv, data_simple_deconv, data_numeric_deconv, type="thickness",
                      conc_element_name="Au", y_max=6, y1_step=0.5, error_max=5) 
