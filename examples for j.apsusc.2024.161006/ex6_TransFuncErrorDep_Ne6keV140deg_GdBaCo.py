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

import os, sys
# changing working directoru to the SpecRec dir
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(parent_dir) 
# Add parent directory to sys.path
sys.path.append(parent_dir)
import numpy as np
import spectraConvDeconv_tools as SCD
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
##################################### PRESET SOME CALC PARAMS  #####################################

# smooth of input spectra with a Savitzky-Golay filter 
SCD.doInputSmooth = True
# the width of the filter window for polynomial fitting, in eV
SCD.filter_window_length = 50

# type of kernel for broadening kernel: gauss, triangle or rectangle
SCD.broadening_kernel_type = "gauss"

# add some noise to the convoluted sim spectrum
SCD.doBroadeningConvNoise = False

# adding gauss noise where noise_power is a gauss sigma
SCD.noise_power = 0.05

SCD.spectrometer_resolutions = [SCD.spectrometer_resolution*1, SCD.spectrometer_resolution*1.05, SCD.spectrometer_resolution*1.1, SCD.spectrometer_resolution*1.2, SCD.spectrometer_resolution*1.5]


SCD.spectrometer_resolution = 0.01

concs_to_calculate = ("Co60","Co40","Co40", "Co20", "Co80")

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
files = os.listdir(spectra_path)

datas = []
legend = []
for file in files:
    for conc in concs_to_calculate:
        if conc in file:
            datas.append(file)
            legend.append(conc.replace("Co","Co=")+"%")
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
    Gd_peak = max(spectrum_int[int((E_peak_Gd-50)/step):int((E_peak_Gd+50)/step)])-spectrum_int[int(E_background/step)]
    Ba_peak = max(spectrum_int[int((E_peak_Ba-50)/step):int((E_peak_Ba+50)/step)])-spectrum_int[int(E_background/step)]
    Co_peak = max(spectrum_int[int((E_peak_Co-50)/step):int((E_peak_Co+50)/step)])-spectrum_int[int(E_background/step)]
    Co_conc =  (Co_peak/crossSection_Co)/((Co_peak/crossSection_Co)+(Gd_peak/crossSection_Gd)+(Ba_peak/crossSection_Ba))

    data_cnv[0,f+1]=Co_conc
    data_simple_deconv[0,f+1]=Co_conc
    data_numeric_deconv[0,f+1]=Co_conc
    
    
    for errorR in range(0, len(SCD.spectrometer_resolutions)):
        
        # do broadening convolution
        conv = SCD.broadening_kernel_convolution(spectrum_int, spectrum_en, SCD.broadening_kernel_type, 
                                                          SCD.spectrometer_resolution)

        Gd_peak = max(conv[int((E_peak_Gd-50)/step):int((E_peak_Gd+50)/step)])-conv[int(E_background/step)]
        Ba_peak = max(conv[int((E_peak_Ba-50)/step):int((E_peak_Ba+50)/step)])-conv[int(E_background/step)]
        Co_peak = max(conv[int((E_peak_Co-50)/step):int((E_peak_Co+50)/step)])-conv[int(E_background/step)]
        conv_Co_conc =  (Co_peak/crossSection_Co)/((Co_peak/crossSection_Co)+(Gd_peak/crossSection_Gd)+(Ba_peak/crossSection_Ba))
        data_cnv[errorR+1, f+1] = conv_Co_conc
        
        # do simple deconvolution
        simple_deconv = SCD.simple_deconvolution(conv)   
        Gd_peak = max(simple_deconv[int((E_peak_Gd-50)/step):int((E_peak_Gd+50)/step)])-simple_deconv[int(E_background/step)]
        Ba_peak = max(simple_deconv[int((E_peak_Ba-50)/step):int((E_peak_Ba+50)/step)])-simple_deconv[int(E_background/step)]
        Co_peak = max(simple_deconv[int((E_peak_Co-50)/step):int((E_peak_Co+50)/step)])-simple_deconv[int(E_background/step)]
        deconv_Co_conc =  (Co_peak/crossSection_Co)/((Co_peak/crossSection_Co)+(Gd_peak/crossSection_Gd)+(Ba_peak/crossSection_Ba))
        data_simple_deconv[errorR+1, f+1] = deconv_Co_conc
        
        #Do more direct deconvolution by solving Fredholm equation with broadening kernel 
        numerical_deconv  = SCD.twomey_deconvolution(conv, spectrum_en, SCD.broadening_kernel_type, SCD.spectrometer_resolutions[errorR])
        Gd_peak = max(numerical_deconv[int((E_peak_Gd-50)/step):int((E_peak_Gd+50)/step)])-numerical_deconv[int(E_background/step)]
        Ba_peak = max(numerical_deconv[int((E_peak_Ba-50)/step):int((E_peak_Ba+50)/step)])-numerical_deconv[int(E_background/step)]
        Co_peak = max(numerical_deconv[int((E_peak_Co-50)/step):int((E_peak_Co+50)/step)])-numerical_deconv[int(E_background/step)]
        numeric_deconv_Co_conc =  (Co_peak/crossSection_Co)/((Co_peak/crossSection_Co)+(Gd_peak/crossSection_Gd)+(Ba_peak/crossSection_Ba))
        data_numeric_deconv[errorR+1, f+1] = numeric_deconv_Co_conc


SCD.calc_name = "ex6_TransFuncErrorDep_Ne6keV140deg_GdBaCo"
save_path = 'out'+os.sep+SCD.calc_name 
if not os.path.exists(save_path):
    os.mkdir(save_path)

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(12, 6))
fig.suptitle(SCD.calc_name + " "+"conc"+" and error in conv and deconv by "+
             SCD.broadening_kernel_type+" kernel spectra"+SCD.logging_options.replace("_",". "))   
        
markers = ['.', 's','*','^','v','>','<','p','x','d','8','1','2']


y1_coeff=100

# Create charts for dependence of the Cr concentration on resolution
for data in range(0, len(datas)):
    ax1.plot(data_cnv[:,0]*100, data_cnv[:,data+1]*y1_coeff, 'k:', marker=markers[data], markersize=5, linewidth=1) 
for data in range(0, len(datas)):
    ax1.plot(data_simple_deconv[:,0]*100, data_simple_deconv[:,data+1]*y1_coeff, 'r--', marker=markers[data], markersize=5, linewidth=1) 
for data in range(0, len(datas)):
    ax1.plot(data_numeric_deconv[:,0]*100, data_numeric_deconv[:,data+1]*y1_coeff,  'b-', marker=markers[data], markersize=5, linewidth=1) 

ax1.set_xlabel('standard deviation of noise additive, %', fontsize=13)
#ax1.set_xticks(np.arange(0, noise_powers[-1]+0.001, 0.001))
ax1.set_ylabel("Co "+'estimated atomic concentration, %', fontsize=13)
ax1.set_yticks(np.arange(0, 101, 10))
ax1.set_ylim(0,85)
# ax1.set_xlim(-0.001, noise_powers[-1]*100+0.1)
ax1.minorticks_on()
ax1.grid()
ax1.text(.96, .98, "A", transform=ax1.transAxes, ha="right", va="top", size=15, weight='bold')
colors = ['black', 'red', 'blue']
line_styles = [':','--','-']
lines = [Line2D([0], [0], color=c, linewidth=1, linestyle='--') for c in colors]
for i in range(3):
    lines[i] = Line2D([0], [0], color=colors[i], linewidth=1, linestyle=line_styles[i]) 
labels = ['convoluted', 'simple deconv', 'numerical deconv']
ax1.legend(lines, labels, frameon=False, bbox_to_anchor=(0.5, 1.08), loc='upper center', ncol=3)


# Create charts for dependence of error on resolution
for data in range(0, len(datas)):
    ax2.plot(data_cnv[1:,0]*100, abs(data_cnv[1:,data+1]-data_cnv[0,data+1])*100, 'k:', marker=markers[data], markersize=5, linewidth=1, label=datas[data].split(".")[0]) 
for data in range(0, len(datas)):
    ax2.plot(data_simple_deconv[1:,0]*100, abs(data_simple_deconv[1:,data+1]-data_simple_deconv[0,data+1])*100, 'r--', marker=markers[data], markersize=5, linewidth=1) 
for data in range(0, len(datas)):
    ax2.plot(data_numeric_deconv[1:,0]*100, abs(data_numeric_deconv[1:,data+1]-data_numeric_deconv[0,data+1])*100,  'b-', marker=markers[data], markersize=5, linewidth=1) 


ax2.set_xlabel('standard deviation of noise additive, %', fontsize=13)
#ax2.set_xticks(np.arange(0, noise_powers[-1]+0.001, 0.001))

ax2.set_ylabel('elemental composition error, %', fontsize=13)

# ax2.set_xlim(0, noise_powers[-1]*100+0.1)
ax2.set_ylim(0,37)
ax2.legend(loc='upper left', fontsize=9)
ax2.minorticks_on()
ax2.grid()
ax2.text(.93, .98,"B", transform=ax2.transAxes, ha="center", va="top", size=15, weight='bold')

#plt.savefig(save_path+os.sep+"noise_conc_dep_"+SCD.calc_name+"_with_"+SCD.broadening_kernel_type+"_kernel"+".png", dpi=300)
plt.show()