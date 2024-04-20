

"""
This program allows to create charts for convolution and deconvolution of
defferent types for one specific simulated or experimental spectrum

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

If you have questions regarding this program, please contact NEEfimov@mephi.ru
"""

import os, time
# changing working directoru to the location of py file
os.chdir(os.path.dirname(os.path.realpath(__file__))) 
import numpy as np
import scipy.signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import spectraConvDeconvLib as SCD
    
##################################### PRESET SOME CALC PARAMS  #####################################

#smooth of input spectra with a Savitzky-Golay filter 
SCD.doInputSmooth = False

#influence smoothing. A window on spectrum points for a 3rd order polynomic fitting 
SCD.filter_window_length = 100

#add some noise to the convoluted sim spectrum
SCD.doBroadeningConvNoise = False

#adding gauss noise where noise_power is a gauss sigma
SCD.noise_power = 0.02

#create gif with animation of broadening convolution
SCD.doAnimation = False

#sigma of gauss kernel for  convolution with constant kernel
SCD.constant_kernel_sigma = 50

# signal-to-noise ratio used in Wiener deconvolution
SCD.SNR = 10

# type of kernel for broadening kernel: gauss, triangle or rectangle
SCD.broadening_kernel_type = "gauss"

# energy resolution (dE/E) of electrostatic energy analyzer for broadening kernel
SCD.spectrometer_resolution = 0.01
#####################################    CHOOSE INPUT FILE    ######################################

# choose one data file
spectrum_path = os.getcwd()+os.sep+"raw_data"+os.sep

# Experimental LEIS spectra. Measured on "Large Mass-Monochromator "MEPhI" facility

#26.10.2023-18-16-Ar+ 18 keV W 60deg 23nA
#spectrum_path += "exp_Ar18keV60deg_WH.dat"

#04.09.2019-14-05-H+ 25 keV 56 nA clean W on glass
#spectrum_path += "exp_H25keV32deg_W.dat"

#29.12.2023-17-16-Ne+ 11 keV  500 nA smart alloy after 1e25 100eV D, another side, 19 uA sputter Ar gun
#spectrum_path += "exp_Ne11keV32deg_WCrY.dat"

#26.12.2023-20-24-Ne+ 130nA smart 1e24 D 100eV +gun
#spectrum_path += "exp_Ne11keV32deg_WCrZr.dat"

#17.05.2018-12-06-Ar+ 18 keV  I~ 175 nA clean SI
#spectrum_path += "exp_Ar18kev_Si.dat"

# Spectra simulated in the SDTrimSP_6.02 code.

#spectrum_path += "sim_Ne11keV32deg_ArKr.dat"

#spectrum_path += "sim_He3keV145deg_Bi2Se3.dat"
#spectrum_path +="sim_Ne6keV140deg_BaCoGd.dat"

#spectrum_path += "sim_Ne11keV32deg_HWCr.dat"
#spectrum_path += "sim_Ne11keV32deg_WCrO.dat"

#spectrum_path +="sim_Ar20keV32deg_HDW.dat"
#spectrum_path += "sim_Ne18keV32deg_HDW.dat"
#spectrum_path += "sim_Ne18keV32deg_HDWthin.dat"
#spectrum_path += "sim_Ar20keV32deg_H10D10W80.dat"
#spectrum_path += "sim_Ne2keV45deg_HW.dat"

spectrum_path += "ex4_sim_Ne2keV45deg_HW"+os.sep+"H50W50.dat"
#spectrum_path += "ex4_sim_Ne2keV45deg_HW"+os.sep+"H40W60.dat"
##################################### GET DATA FROM INPUT FILE #####################################

SCD.calc_name = spectrum_path.split(os.sep)[-1].split(".")[0]
SCD.Emin = 100
spectrum_en, spectrum_int = SCD.import_data(spectrum_path)

# or test on input analytical specific curves instead of external spectrum_file
do_gausses = False

# 1 two triangles
"""
spectrum_int = np.zeros(len(spectrum_int))
for i in range (int((1500-25)/step), int(1500/step)): spectrum_int[i] = (i-(int((1500-25)/step)))/(int(25/step))
for i in range (int(1500/step), int((1500+25)/step)): spectrum_int[i] = (int((1500+25)/step)-i)/(int(25/step))
for i in range (int((10500-250)/step), int(10500/step)): spectrum_int[i] = (i-(int((10500-250)/step)))/(int(250/step))
for i in range (int(10500/step), int((10500+250)/step)): spectrum_int[i] = (int((10500+250)/step)-i)/(int(250/step))
SCD.calc_name = "sim_triangles"
"""

# 2 several gausses
if do_gausses:
    local_sigma = 120
    SCD.Emax = 30000
    spectrum_en = np.arange(1, SCD.Emax, SCD.step)
    spectrum_int = np.zeros(len(spectrum_en))
    peaks_num = 0
    for energy in range(1000, SCD.Emax, 1000):
        peaks_num+=1
        spectrum_int+=np.exp(-(spectrum_en-energy)**2/2/local_sigma**2)
    SCD.calc_name = "sim_sev_gausses_sigma="+str(local_sigma)

# 3 rectangular pulse
"""
spectrum_int = np.zeros(len(spectrum_int))
for i in range (int(5000/step),int(10000/step)): spectrum_int[i] = 1
calc_name = "sim_rectangular_pulses"
"""

######################## DO CONSTANT KERNEL  CONVOLUTION AND DECONVOLUTION  ########################

# create gauss kernel for convolution with constant kernel
gauss_kernel = np.exp(-(spectrum_en-spectrum_en.mean())**2/2/SCD.constant_kernel_sigma**2)
#remove low values to speed up the calculation and reduce output array's size
gauss_kernel = gauss_kernel[gauss_kernel>.001]

t1 = time.time()
basic_gauss_convolution = SCD.norm(SCD.basic_convolution(spectrum_int, gauss_kernel))
t2 = time.time()
print ("Basic convolution time, ms: "+str((t2-t1)*1000))

# Alternatively we can use scipy's embedded convolution, which is faster than our 
# basic_convolution function, as scipy.signal.convolve uses Laplace transform and FFT
# "same" mode is to prevent spectrum shifting after the convolution
t1 = time.time()
scipy_convolution = SCD.norm(scipy.signal.convolve(spectrum_int, gauss_kernel, mode='same'))
t2 = time.time()
print ("Scipy convolution time, ms: "+str((t2-t1)*1000))

deconvolution_of_basic_conv = SCD.norm(SCD.wiener_deconvolution(basic_gauss_convolution, gauss_kernel))

######################  DO BROADENING CONVOLUTION AND DECONVOLUTION   ##############################

isExp = False
if "exp" in SCD.calc_name: isExp=True

def deconv (signal):
    # Do our simple deconvolution
    # Actually it is a division of broadening_gauss_convolution (as measured by electrostatic analyzer) 
    # energy spectrum by the energy. For more info see Urusov's papers: https://doi.org/10.1134/1.1258598
    simple_deconv = SCD.simple_deconvolution(signal)
    # Do more direct deconvolution by solving Fredholm equation with broadening kernel 
    t1 = time.time()
    numerical_deconv  = SCD.norm(SCD.twomey_deconvolution(signal, spectrum_en, SCD.broadening_kernel_type, SCD.spectrometer_resolution))
    t2 = time.time()
    print ("Broadening "+SCD.broadening_kernel_type+" deconvolution time, s: "+str((t2-t1)))
    return simple_deconv, numerical_deconv

def conv (signal):
    t1 = time.time()
    broadening_gauss_convolution = SCD.broadening_kernel_convolution(signal, spectrum_en,SCD.broadening_kernel_type, SCD.spectrometer_resolution)
    t2 = time.time()
    print ("Broadening "+SCD.broadening_kernel_type+" convolution time, s: "+str((t2-t1)))
    return broadening_gauss_convolution

if isExp:
    simple_deconv, numerical_deconv = deconv(spectrum_int)
    broadening_convolution_of_simple_deconv = conv(simple_deconv)
    broadening_convolution_of_numeric_deconv =conv(numerical_deconv)
    signals = [basic_gauss_convolution, simple_deconv, numerical_deconv]
    labels = ['Basic gauss convolution ', 'Simple deconvolution', 'Deconvolution by solving Fredholm integral equation']
    
else:
    broadening_sim_convolution = conv(spectrum_int)
    simple_deconv, numerical_deconv = deconv(broadening_sim_convolution)
    signals = [basic_gauss_convolution, broadening_sim_convolution, simple_deconv, numerical_deconv]
    labels = ['Basic gauss convolution ', 'Broadening '+SCD.broadening_kernel_type+' convolution, R= '+str(SCD.spectrometer_resolution),
              'Simple deconvolution', 'Deconvolution by solving Fredholm integral equation']

################################     VISUALIZE DATA IN WEB BROWSER     #############################

fig = make_subplots(rows = len(signals), x_title = 'Energy, eV', y_title = 'Intensity, r.u.', 
                    shared_xaxes = True, vertical_spacing = 0.01)

for row_id in range (0,len(signals)):
    fig.add_trace(
        go.Scatter(
            x = spectrum_en,
            y = signals[row_id], 
            name = labels[row_id],
            showlegend=True,
            legendgroup = labels[row_id],
            line_color="#ff2a1b"
        ),
        row = row_id+1,
        col = 1,
    )

for row_id in range (1,len(signals)+1):
    fig.add_trace(
        go.Scatter(
            x = spectrum_en,
            y = spectrum_int, 
            name = "Raw spectrum",
            line={'width': 1, 'dash':'dot'},
            line_color="#000000",
            legendgroup = labels[row_id-1]
        ),
        row = row_id,
        col = 1,
    )
    

fig.add_trace(
        go.Scatter(
            x = spectrum_en,
            y = gauss_kernel, 
            name = 'Gauss_kernel',
            legendgroup = labels[0],
            line_color="#FB6000"
        ),
        row = 1,
        col = 1,
    ) 

fig.add_trace(
        go.Scatter(
            x = spectrum_en,
            y = deconvolution_of_basic_conv, 
            name = 'Wiener deconvolution',
            legendgroup = labels[0],
            line_color="#0000FF"
        ),
        row = 1,
        col = 1,
    ) 

if isExp:
    fig.add_trace(
            go.Scatter(
                x = spectrum_en,
                y = broadening_convolution_of_simple_deconv, 
                name = 'Broadening conv of simple deconv, R= '+str(SCD.spectrometer_resolution),
                legendgroup = labels[1],
                line_color="#0000FF"
            ),
            row = 2,
            col = 1,
        ) 
    fig.add_trace(
            go.Scatter(
                x = spectrum_en,
                y = broadening_convolution_of_numeric_deconv, 
                name = 'Broadening conv of numeric deconv, R= '+str(SCD.spectrometer_resolution),
                legendgroup = labels[2],
                line_color="#0000FF"
            ),
            row = 3,
            col = 1,
        ) 
   
    
fig.update_layout(title_text="Demonstration of "+SCD.calc_name+" energy spectrum Convolution/Deconvolution with constant and broadening "+
                  SCD.broadening_kernel_type+" kernel"+SCD.logging_options,
                  legend_tracegroupgap=350, yaxis=dict(range=[0, 1]))
fig.update_yaxes(range=[0, 1])
fig.show()
    
#####################################     SAVE PIC TO OUT DIR    ###################################
if "ex" in spectrum_path and "exp" not in spectrum_path:
    save_path = 'out'+os.sep+spectrum_path.split(os.sep)[-2]+os.sep+SCD.calc_name
else:
    save_path = 'out'+os.sep+SCD.calc_name
if not os.path.exists(save_path):
    os.mkdir(save_path)


if not isExp:
    plt.plot(spectrum_en/1000, spectrum_int, 'b-',linewidth=1.5, label='Raw spectrum', alpha=0.7) 
    plt.plot(spectrum_en/1000, broadening_sim_convolution[0:len(spectrum_en)], 'k--',linewidth=2.5, alpha=0.9, 
            label='Convoluted with dE/E='+str(SCD.spectrometer_resolution)) 
    plt.plot(spectrum_en/1000, simple_deconv[0:len(spectrum_en)], 'r:', linewidth=2.5, alpha=0.7, label='Simple Deconvolution') 
    plt.plot(spectrum_en/1000, numerical_deconv[0:len(spectrum_en)], 'g-.', linewidth=2.5, alpha=0.85, label='Numerical Deconvolution') 
    plt.legend(fontsize=11,loc='upper center')
    plt.ylim(0,1.1)
    plt.xlim(0, spectrum_en[-1]/1000+0.2)
    #plt.xticks(np.arange(1, spectrum_en[-1]/1000, ))
    plt.minorticks_on()
    plt.xlabel('energy, keV')
    plt.ylabel('intensity, r.u.')
    plt.title("Energy spectra of "+SCD.calc_name+". "+SCD.logging_options.replace("_",""), y=1.02)
    plt.savefig(save_path+os.sep+"spec_reconstr_"+SCD.calc_name+"_with_"+SCD.broadening_kernel_type+"_kernel"+SCD.logging_options+".png", dpi=400)
    #plt.show()
if isExp:
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True, figsize=(12, 6))
    ax1.plot(spectrum_en/1000, spectrum_int, 'b--',linewidth=1, label='Raw spectrum') 
    ax1.plot(spectrum_en/1000, simple_deconv[0:len(spectrum_en)], 'r:', linewidth=1, label='Simple Deconvolution') 
    ax1.plot(spectrum_en/1000, broadening_convolution_of_simple_deconv[0:len(spectrum_en)], 'k-',linewidth=1,
            label='Convolution of simple deconvolution with dE/E='+str(SCD.spectrometer_resolution)) 
    ax2.plot(spectrum_en/1000, spectrum_int, 'b--',linewidth=1, label='Raw spectrum') 
    ax2.plot(spectrum_en/1000, numerical_deconv[0:len(spectrum_en)], 'r:', linewidth=1, label='Numerical Deconvolution') 
    ax2.plot(spectrum_en/1000, broadening_convolution_of_numeric_deconv[0:len(spectrum_en)], 'k-',linewidth=1,
            label='Convolution of numerical deconvolution with dE/E='+str(SCD.spectrometer_resolution)) 
    ax1.set_ylim(0,1)
    ax1.set_xlim(0, spectrum_en[-1]/1000+0.1)
    ax1.set_xticks(np.arange(0, spectrum_en[-1]/1000+1, 1))
    ax1.minorticks_on()
    ax1.legend(fontsize=9)
    ax2.legend(fontsize=9)
    fig.suptitle("Energy spectra of "+SCD.calc_name+". "+SCD.logging_options.replace("_",""))
    fig.savefig(save_path+os.sep+"spec_reconstr_"+SCD.calc_name+"_with_"+SCD.broadening_kernel_type+"_kernel"+SCD.logging_options+".png", dpi=400)
    #plt.show()

#####################################     SAVE DATA TO OUT DIR    ##################################

with open(save_path+os.sep+"spec_reconstr_"+SCD.calc_name+"_with_"+SCD.broadening_kernel_type+"_kernel"+SCD.logging_options+".dat", "w",newline='\n') as f:   
    f.write("Energy,keV Raw_signal Basic_conv,sigma="+str(SCD.constant_kernel_sigma)+
            " Wiener_deconv")
    if isExp:
        f.write(" Simple_deconv Numeric_deconv Broaden_"+SCD.broadening_kernel_type+"_simple_conv,dE/E="+str(SCD.spectrometer_resolution)
        +" Broaden_"+SCD.broadening_kernel_type+"_numeric_conv,dE/E="+str(SCD.spectrometer_resolution)+"\n")
    else:
        f.write(" Broaden_"+SCD.broadening_kernel_type+"_simple_conv,dE/E="+str(SCD.spectrometer_resolution)+
                " simple_deconv numeric_deconv"+"\n")
        
    for i in range (len(spectrum_en)):
        f.write(str("{:.3e}".format(spectrum_en[i]/1000)).rjust(14)+" "+
                str("{:.3e}".format(spectrum_int[i])).rjust(14)+" "+
                str("{:.3e}".format(basic_gauss_convolution[i])).rjust(14)+" "+
                str("{:.3e}".format(deconvolution_of_basic_conv[i])).rjust(14)+" ")
        if isExp:
            f.write(str("{:.3e}".format(simple_deconv[i])).rjust(14)+" "+
                    str("{:.3e}".format(numerical_deconv[i])).rjust(14)+" "+
                    str("{:.3e}".format(broadening_convolution_of_simple_deconv[i])).rjust(14)+" "+
                    str("{:.3e}".format(broadening_convolution_of_numeric_deconv[i])).rjust(14)+"\n")
        else:
            f.write(str("{:.3e}".format(broadening_sim_convolution[i])).rjust(14)+" "+
                    str("{:.3e}".format(simple_deconv[i])).rjust(14)+" "+
                    str("{:.3e}".format(numerical_deconv[i])).rjust(14)+"\n")
    f.close
    
#####################  CREATE ANIMATED CHARTS #####################

if SCD.doAnimation:
    SCD.create_animated_chart(spectrum_int, spectrum_en)

#####################  CREATE PEAK INT DEPENDENCIES ON ENERGY #####################

if do_gausses:
    save_path = 'out'+os.sep+"sim_sev_gausses"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    gauss_energy = np.zeros(peaks_num+1)
    gauss_sim_deconv_intensity_dep = np.zeros(peaks_num+1)
    gauss_num_deconv_intensity_dep = np.zeros(peaks_num+1)
    gauss_conv_intensity_dep = np.zeros(peaks_num+1)
    gauss_conv_area_dep = np.zeros(peaks_num+1)

    gauss_raw_width_dep = np.zeros(peaks_num+1)
    gauss_conv_width_dep = np.zeros(peaks_num+1)

    for peak in range (1, peaks_num, 1):
        gauss_sim_deconv_intensity_dep[peak] = max(simple_deconv[int((peak*1000-500)/SCD.step):int((peak*1000+500)/SCD.step)])
        gauss_num_deconv_intensity_dep[peak] = max(numerical_deconv[int((peak*1000-500)/SCD.step):int((peak*1000+500)/SCD.step)])
        gauss_conv_intensity_dep[peak]=max(broadening_sim_convolution[int((peak*1000-500)/SCD.step):int((peak*1000+500)/SCD.step)])
        gauss_energy[peak] = peak
        border = False
        border2=False
        for i in range (int((peak*1000-500)//SCD.step),int((peak*1000+500)//SCD.step),1):
            gauss_conv_area_dep[peak]+=broadening_sim_convolution[i]
            if (broadening_sim_convolution[i]> gauss_conv_intensity_dep[peak]/2 and not border):
                gauss_conv_width_dep[peak]=i*SCD.step
                border = True
            if (broadening_sim_convolution[i]< gauss_conv_intensity_dep[peak]/2 and border):
                gauss_conv_width_dep[peak]=i*SCD.step-gauss_conv_width_dep[peak]
                border = False
                
            if (spectrum_int[i]> 1/2 and not border2):
                gauss_raw_width_dep[peak]=i*SCD.step
                border2 = True
            if (spectrum_int[i]< 1/2 and border2):
                gauss_raw_width_dep[peak]=i*SCD.step-gauss_raw_width_dep[peak]
                border2 = False
        gauss_conv_width_dep[peak]/=gauss_raw_width_dep[peak]
        gauss_conv_width_dep[peak]-=1
        
    with open(save_path+os.sep+SCD.calc_name+"_dEtoE="+str(SCD.spectrometer_resolution)+".dat", "w",newline='\n') as f:   
        f.write("Energy,keV ConvInt convWidth convArea NumDeconvInt SimDeconvInt"+"\n")
        for i in range (1, peaks_num+1, 1):
            f.write(str("{:.3e}".format(gauss_energy[i]).rjust(10))+" "+
                    str("{:.3e}".format(gauss_conv_intensity_dep[i])).rjust(10)+" "+
                    str("{:.3e}".format(gauss_conv_width_dep[i])).rjust(10)+" "+
                    str("{:.3e}".format(gauss_conv_area_dep[i])).rjust(10)+" "+
                    str("{:.3e}".format(gauss_sim_deconv_intensity_dep[i])).rjust(10)+" "+
                    str("{:.3e}".format(gauss_num_deconv_intensity_dep[i])).rjust(10)+"\n")
        f.close          

    gauss_conv_area_dep /= max(gauss_conv_area_dep) 
    plt.show()
    plt.title('Demonstration of  distortion and reconstruction of \nGaussian peaks sigma='+str(local_sigma)+' and dE/E='+str(SCD.spectrometer_resolution), fontsize=10)
    plt.plot(gauss_energy[1:peaks_num], gauss_conv_intensity_dep[1:peaks_num],'*k-.',  label="conv. intensity", linewidth=1.5) 
    plt.plot(gauss_energy[1:peaks_num], gauss_conv_width_dep[1:peaks_num], '.r:',  label="conv. width",linewidth=1.5) 
    plt.plot(gauss_energy[1:peaks_num], gauss_conv_area_dep[1:peaks_num], 'sg-',  label="conv. area", linewidth=1.5) 
    plt.plot(gauss_energy[1:peaks_num], gauss_num_deconv_intensity_dep[1:peaks_num], '^b-',  label="num. deconv. intensity", linewidth=1.5) 
    plt.plot(gauss_energy[1:peaks_num], gauss_sim_deconv_intensity_dep[1:peaks_num], 'vm:',  label="simple deconv. intensity", linewidth=1.5) 
    plt.legend( frameon=False, loc='lower right', fontsize=9)
    plt.xlim(0, spectrum_en[-1]/1000-2)
    plt.xticks(np.arange(0, spectrum_en[-1]/1000+1, 5))
    plt.ylim(0,1)
    plt.minorticks_on()
    plt.xlabel('energy, keV')
    plt.ylabel('intensity, a.u.')   
    plt.grid(True, linestyle=':') 
    plt.savefig(save_path+os.sep+SCD.calc_name+"_dEtoE="+str(SCD.spectrometer_resolution)+".png", dpi=300)
    plt.show()

