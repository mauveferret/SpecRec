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
SCD.filter_window_length = 20

#add some noise to the convoluted sim spectrum
SCD.doBroadeningConvNoise = False
#adding gauss noise where noise_power is a gauss sigma
SCD.noise_power = 0.02

# positions of elastic peaks: Scattering of Ar in W and recoil of W in eV
E_peak_H = 180
E_peak_W = 1875

# detection energy widths  for the specific angle detection width (dBeta=4 deg)
E_width_H = 192-167
E_width_W = 1885 - 1865

# energy where only background is seen, that can be subtracted from H an D peaks
E_background_of_H = 215

#  elemental sensitivity in a form of difference of squares of impact parameters 
# for a specific registration solid angle, Å^2
crossSection_H = 0.0099122
crossSection_W = 0.0243854*0.5

SCD.spectrometer_resolutions = (  0.002, 0.005, 0.008, 0.01, 0.012,0.015, 0.02, 0.025, 0.03, 0.035)

concs_to_calculate = ("H20","H40","H60", "H80")


#####################################    CHOOSE INPUT FILES    ######################################

# choose data files
spectra_path = os.getcwd()+os.sep+"raw_data"+os.sep+"ex4_sim_Ne2keV45deg_HW"

SCD.calc_name = str(spectra_path.split(os.sep)[-1])
files = os.listdir(spectra_path)
datas = []
legend = []
for file in files:
    for conc in concs_to_calculate:
        if conc in file:
            datas.append(file)
            legend.append(conc.replace("H","H=")+"%")
            break;

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
    W_peak = max(spectrum_int[int((E_peak_W-E_width_W/2)/step):int((E_peak_W+E_width_W/2)/step)])
    without_background = spectrum_int - spectrum_int[int(E_background_of_H/step)]
    H_peak = max(without_background[int((E_peak_H-E_width_H/2)/step):int((E_peak_H+E_width_H/2)/step)])
    real_H_conc =  (H_peak/crossSection_H)/((H_peak/crossSection_H)+(W_peak/crossSection_W))
        
    data_cnv[0,f+1]=real_H_conc
    data_simple_deconv[0,f+1]=real_H_conc
    data_numeric_deconv[0,f+1]=real_H_conc
    
    for R in range(0,len(SCD.spectrometer_resolutions)):
        
        # do broadening convolution
        conv = SCD.broadening_kernel_convolution(spectrum_int, spectrum_en, SCD.broadening_kernel_type, 
                                                          SCD.spectrometer_resolutions[R])
        W_peak = max(conv[int((E_peak_W-E_width_W/2)/step):int((E_peak_W+E_width_W/2)/step)])  
        without_background = conv - conv[int(E_background_of_H/step)]
        H_peak = max(without_background[int((E_peak_H-E_width_H/2)/step):int((E_peak_H+E_width_H/2)/step)])
        conv_H_conc =  (H_peak/crossSection_H)/((H_peak/crossSection_H)+(W_peak/crossSection_W))
        data_cnv[R+1, f+1] = conv_H_conc
        
        # do simple deconvolution
        simple_deconv = SCD.simple_deconvolution(conv)
        W_peak = max(simple_deconv[int((E_peak_W-E_width_W/2)/step):int((E_peak_W+E_width_W/2)/step)])
        without_background = simple_deconv - simple_deconv[int(E_background_of_H/step)]
        H_peak = max(without_background[int((E_peak_H-E_width_H/2)/step):int((E_peak_H+E_width_H/2)/step)])
        deconv_H_conc =  (H_peak/crossSection_H)/((H_peak/crossSection_H)+(W_peak/crossSection_W))
        data_simple_deconv[R+1, f+1] = deconv_H_conc
        
        #Do more direct deconvolution by solving Fredholm equation with broadening kernel 
        numerical_deconv  = SCD.twomey_deconvolution(conv, spectrum_en, SCD.broadening_kernel_type, 
                                                              SCD.spectrometer_resolutions[R])
        W_peak = max(numerical_deconv[int((E_peak_W-E_width_W/2)/step):int((E_peak_W+E_width_W/2)/step)])  
        without_background = numerical_deconv - numerical_deconv[int(E_background_of_H/step)]
        H_peak = max(without_background[int((E_peak_H-E_width_H/2)/step):int((E_peak_H+E_width_H/2)/step)])
        numeric_deconv_H_conc =  (H_peak/crossSection_H)/((H_peak/crossSection_H)+(W_peak/crossSection_W))
        data_numeric_deconv[R+1, f+1] = numeric_deconv_H_conc

#save data to output and create plots
SCD.save_conc_tables(datas, data_cnv, data_simple_deconv, data_numeric_deconv)
SCD.create_conc_plots(legend, data_cnv, data_simple_deconv, data_numeric_deconv, conc_element_name="H", y_max=90, error_max=48) 
