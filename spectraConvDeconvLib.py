import os
import numpy as np
from numpy.fft import fft, ifft   
from inteq import SolveFredholm 
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

#smooth of input spectra with a Savitzky-Golay filter 
doInputSmooth = False

# the width of the filter window for polynomial fitting, in eV
filter_window_length = 100

#add some noise to the convoluted sim spectrum
doBroadeningConvNoise = False
#adding gauss noise where noise_power is a gauss sigma
noise_power = 0.02

#create gif with animation of broadening convolution
doAnimation = False

# energy step for final interpolated energy axis
step = 2 # in eV

Emin = 500
Emax = 20000

#sigma of gauss kernel for  convolution with constant kernel
constant_kernel_sigma = 50

# signal-to-noise ratio used in Wiener deconvolution
SNR = 10

# type of kernel for broadening kernel: gauss, triangle or rectangle
broadening_kernel_type = "gauss"

# energy resolution (dE/E) of electrostatic energy analyzer for broadening kernel
spectrometer_resolutions = ( 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05)
spectrometer_resolution = 0.01

# name for output
calc_name = ""
# additional info on the study for the iotput filenames and titles
logging_options = ""

######################################    SOME FUNCTIONS     #######################################

# creates array, representing kernel with arbitrary FWHM and form
# FWHM is full width at half maximum
def broadening_kernel(type, FWHM):
    if type=="gauss":
        def broadening_gauss_kernel(energies, shift):
            return np.exp(-((energies-shift)**2)/(2*(FWHM/2*shift)**2)) 
        return broadening_gauss_kernel
    
    elif type=="rectangle":
        def broadening_rectangle_kernel(energies, shift):
            if  hasattr(shift, "__len__"):
                #this part is for Fredholm solver
                energies = np.hstack(energies)
                rectangle_kernel = np.zeros((len(energies), len(shift)))
                for j in range (len(shift)):
                    for i in range(len(energies)):
                        rectangle_kernel[i][j] = 1 if np.abs(energies[i]-shift[j])<FWHM/2*energies[i] else 0
                return rectangle_kernel   
            else:
                # this part for the constant input shift
                rectangle_kernel = np.zeros(len(energies))
                for i in range(len(energies)):
                    rectangle_kernel[i] = 1 if np.abs(energies[i]-shift)<FWHM/2*energies[i] else 0
                return rectangle_kernel
        return broadening_rectangle_kernel
    
    elif type=="triangle":
        def broadening_triangle_kernel(energies, shift):
            if  hasattr(shift, "__len__"):
                #this part is for Fredholm solver
                energies = np.hstack(energies)
                triangle_kernel = np.zeros((len(energies), len(shift)))
                for j in range (len(shift)):
                    for i in range(len(energies)):
                        if (abs(energies[i]-shift[j])<=FWHM*shift[j]):
                            if (energies[i]<=shift[j]):
                                triangle_kernel[i][j]=(energies[i]-shift[j]+FWHM*shift[j])/(FWHM*shift[j])
                            if (energies[i]>shift[j]):
                                triangle_kernel[i][j]=(shift[j]+FWHM*shift[j]-energies[i])/(FWHM*shift[j])
                        
                return triangle_kernel   
            else:
                # this part for the constant input shift
                triangle_kernel = np.zeros(len(energies))
                for i in range(len(energies)):
                    if (abs(energies[i]-shift)<=FWHM*shift):
                        if (energies[i]<=shift):
                            triangle_kernel[i]=(energies[i]-shift+FWHM*shift)/(FWHM*shift)
                        if (energies[i]>shift):
                            triangle_kernel[i]=(shift+FWHM*shift-energies[i])/(FWHM*shift)
                return triangle_kernel
        return broadening_triangle_kernel
    
    #example of the use of  experimentally measured transmission function 
    # 11.10.2022-19-44-Ne+ 13kev measured on "MEPhI" Large Mass Monochromator"
    elif type=="LMM":      
        def broadening_exp_kernel(energies, shift):  
            
            # TODO finish it
            
            #this part is better to be transferred out of the broadening_kernel function
            kernel_path = os.getcwd()+os.sep+"raw_data"+os.sep+"exp_kernel_example_LMM.txt"
            kernel_file = open(kernel_path).read()
            lines = kernel_file.splitlines()
            raw_kernel_int = np.zeros(len(lines))
            raw_kernel_en = np.zeros(len(lines))
            for i in range(0, len(lines)):
                lines[i] = lines[i].replace('\t'," ").replace(",", ".").replace("E","e")
                data = lines[i].split(" ")
                raw_kernel_en[i] = float(data[0])
                raw_kernel_int[i] = float(data[1]) 
            spectrum_en = np.arange(0, raw_kernel_en[-1], step) 
            raw_kernel_en-=12800
            raw_kernel_en/=128
            raw_kernel_en*=FWHM
            raw_kernel_en+=shift
            kernel_int = np.interp(spectrum_en,raw_kernel_en, raw_kernel_int)
            kernel_int = np.interp(energies,spectrum_en, kernel_int)
            kernel_int /= 8.7E-7
            return kernel_int
        return broadening_exp_kernel


######################################     CONVOLUTIONS     #######################################


# based on https://inclab.ru/profi/realizaciya-svertki-v-scilab
def basic_convolution(f,g):
    Lf = len(f)
    Lg = len(g)
    L = Lf + Lg - 1
    cnv = np.zeros(L)
    for k in range(L): 
        for j in range  (max(0, k-int(Lg/2) + 1), min(k+int(Lg/2), Lf)):      
            cnv[k] += f[j] * g[k+int(Lg/2) - j]
    return cnv


# "convolution" with gauss kernel, whose sigma depends on energy 
# (as do transmission function of electrostatic spectrometers)
def broadening_kernel_convolution(f, raw_en, kernel_type = broadening_kernel_type, deltaEtoE=spectrometer_resolution, step=step):
    #we create this gauss array only to estimate max vicinity (energy width) we need to 
    # analyse during convolution for each point of raw signal
    g = broadening_kernel(kernel_type, deltaEtoE)(raw_en,len(raw_en)*step)
    g = g[g>.001]
    Lg = len(g)
    Lf = len(f)
    en = np.arange(raw_en[0], (len(raw_en)+len(g))*step, step)
    L = len(en)
    cnv = np.zeros(L)
    for k in range (1, L):
        g = broadening_kernel(kernel_type, deltaEtoE)(raw_en,en[k])
        for i in range(max(0, k + 1 - Lg), min(k+Lg, Lf)):
            cnv[k] += f[i] * g[i]
    if doBroadeningConvNoise:
        #adding noise to the spectrum
        cnv+=np.random.normal(0,noise_power, len(cnv))
    return cnv

####################################     DECONVOLUTIONS     #######################################

# based on https://gist.github.com/danstowell/f2d81a897df9e23cc1da
def wiener_deconvolution(signal, kernel, SNR=SNR):
    sequence  = np.hstack((kernel , np.zeros(len(signal) - len(kernel)))) 
    H = fft(sequence ) 
    deconvolved = np.real(ifft(fft(signal)*np.conj(H)/(H*np.conj(H) + SNR**2))) 
    
    for i in range (len(deconvolved)-1, int(len(kernel)/2), -1):
        deconvolved[i] = deconvolved[i-int(len(kernel)/2)]
    return deconvolved


# method proposed in Zhabrev, G. I., & Zhdanov, S. K. (1979). 
# Restoration of a real energy distribution of particles passed through a spectrometer with 
# a given instrument function. Zhurnal Tekhnicheskoj Fiziki, 49(11), 2450–2454.
def analitycal_deconvolution(signal, en, deltaEtoE=spectrometer_resolution):
    step = en[2]-en[1]
    shifted_signal = np.zeros(len(signal))
    for E in range (0,len(signal)-int((en[-1]*deltaEtoE)/step)-10):
        shifted_signal[E]=signal[int(E*(1+deltaEtoE))]
    sig_diff=np.diff(shifted_signal)
    deconvoluted = np.zeros(len(signal))    
    for E in range (1, len(deconvoluted)-2):
        deconvoluted[E] = shifted_signal[E]/(deltaEtoE*en[E])-sig_diff[E]
    return deconvoluted


#based on PhD dissertation of Urusov V.A.
def simple_deconvolution(signal):
    simple_deconv = np.zeros(len(signal))
    for i in range (1,len(signal)):
        simple_deconv[i] = signal[i]/(step*i)
    return(simple_deconv)


# numerical solution of Fredholm integral equations of the first kind
# based on https://doi.org/10.1145/321150.321157
def  twomey_deconvolution(signal, spectrum_en, kernel_type = broadening_kernel_type, deltaEtoE = spectrometer_resolution, num=1000):
    # define function to deconvolve
    def g(s):
        #Fredholm solver call this function with some s array, whose length can not be as high 
        #as for real spectra due to long computation, so we need to provide interpolation
        local_energies = np.arange(0, len(signal), 1)
        f = np.interp(s,local_energies, signal)
        return f
    # apply the solver
    # larger parameter num may improve the solution: num=int(len(signal)//step)
    spectrum_en_deconv, numerical_deconv_local = SolveFredholm(broadening_kernel(kernel_type, deltaEtoE), 
                                                               g, a=1, b=len(signal), gamma=1, num=num)
    spectrum_en_deconv *=step
    numerical_deconv = np.interp(spectrum_en,spectrum_en_deconv, numerical_deconv_local)
    return numerical_deconv


####################################     GENERAL FUNCTIONS     #####################################


def import_data(spectrum_file):  
    global logging_options
    if doInputSmooth and ("_Smooth="+str(filter_window_length)+" eV" not in logging_options):
        logging_options+="_Smooth="+str(filter_window_length)+" eV"
    if doBroadeningConvNoise  and ("_Noise="+str(noise_power)+"_"  not in logging_options):
        logging_options+="_Noise="+str(noise_power)+"_" 
        
    lines = spectrum_file.splitlines()
    raw_spectrum_int = np.zeros(len(lines))
    raw_spectrum_en = np.zeros(len(lines))

    for i in range(1, len(lines)):
        lines[i] = lines[i].replace('\t'," ").replace(",", ".").replace("E","e")
        data = lines[i].split(" ")
        raw_spectrum_en[i] = float(data[0])
        raw_spectrum_int[i] = float(data[1])

    # do interpolation with new energy step and normalization to 1 in range (Emin, Emax)
    spectrum_en = np.arange(0, raw_spectrum_en[-1], step)
    #scaling range in eV (influence spectra normalization in Web charts and output files)
    spectrum_int = np.interp(spectrum_en,raw_spectrum_en, raw_spectrum_int)
    global Emax
    Emax = spectrum_en[-1]-100

    if doInputSmooth:
        spectrum_int = scipy.signal.savgol_filter(spectrum_int, int(filter_window_length/step), 3)
    
    # spectrum normalization to 1 in range (Emin, Emax)
    spectrum_int = norm(spectrum_int) 
    return spectrum_en, spectrum_int


# spectrum normalization to 1 in range (Emin, Emax)
def norm(arr):
    return arr/max(arr[int(Emin/step):int(Emax/step)])

def peak(arr):
    return max(arr[int(Emin/step):int(Emax/step)])
    
#################################   SAVE TO OUTPUT AND CREATE PLOTS  ###############################


# save results of dep of conc and thickness on resolution to file
def save_conc_tables(datas, data_cnv, data_simple_deconv, data_numeric_deconv):
    save_path = 'out'+os.sep+calc_name
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with open(save_path+os.sep+"conc_conv_"+calc_name+"_with_"+broadening_kernel_type+"_kernel"+logging_options.replace(" ","")+".dat", "w",newline='\n') as f:   
        f.write(("resolution").ljust(14))
        for data in range(0, len(datas)):
                f.write((str(datas[data].split(".")[0])).ljust(14))
        f.write("\n")
        for R in range(0, len(spectrometer_resolutions)+1):
            for data in range(0, len(datas)+1):
                f.write((str("{:.3e}".format(data_cnv[R,data])+"")).ljust(14)+" ")
            f.write("\n")
            
    with open(save_path+os.sep+"conc_simple_deconv_"+calc_name+"_with_"+broadening_kernel_type+"_kernel"+logging_options+".dat", "w",newline='\n') as f:   
        f.write(("resolution").ljust(14))
        for data in range(0, len(datas)):
                f.write((str(datas[data].split(".")[0])).ljust(14))
        f.write("\n")
        for R in range(0, len(spectrometer_resolutions)+1):
            for data in range(0, len(datas)+1):
                f.write((str("{:.3e}".format(data_simple_deconv[R,data])+"")).ljust(14)+" ")
            f.write("\n")
            
    with open(save_path+os.sep+"conc_numeric_deconv_"+calc_name+"_with_"+broadening_kernel_type+"_kernel"+logging_options+".dat", "w",newline='\n') as f:   
        f.write(("resolution").ljust(14))
        for data in range(0, len(datas)):
                f.write((str(datas[data].split(".")[0])).ljust(14))
        f.write("\n")
        for R in range(0, len(spectrometer_resolutions)+1):
            for data in range(0, len(datas)+1):
                f.write((str("{:.3e}".format(data_numeric_deconv[R,data])+"")).ljust(14)+" ")
            f.write("\n")
       
            
def create_conc_plots(datas, data_cnv, data_simple_deconv, data_numeric_deconv, type = "conc", conc_element_name="Co", y_max=101, y1_step=10, error_max=60):   
    save_path = 'out'+os.sep+calc_name 
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(12, 6))
    fig.suptitle(calc_name + " "+type+" and error in conv and deconv by "+broadening_kernel_type+" kernel spectra"+logging_options.replace("_",". "))   
            
    markers = ['.', 's','*','^','v','>','<','p','x','d','8','1','2']
    
    y1_coeff = 1
    if type=="conc":
        y1_coeff*=100
    
    # Create charts for dependence of the Cr concentration on resolution
    for data in range(0, len(datas)):
        ax1.plot(data_cnv[:,0], data_cnv[:,data+1]*y1_coeff, 'k:', marker=markers[data], markersize=5, linewidth=1) 
    for data in range(0, len(datas)):
        ax1.plot(data_simple_deconv[:,0], data_simple_deconv[:,data+1]*y1_coeff, 'r--', marker=markers[data], markersize=5, linewidth=1) 
    for data in range(0, len(datas)):
        ax1.plot(data_numeric_deconv[:,0], data_numeric_deconv[:,data+1]*y1_coeff,  'b-', marker=markers[data], markersize=5, linewidth=1) 

    ax1.set_xlabel('relative resolution of spectrometer, ΔE/E', fontsize=13)
    ax1.set_xticks(np.arange(0, spectrometer_resolutions[-1]+0.01, 0.01))
    ax1.set_ylabel(conc_element_name+' estimated atomic concentration, %', fontsize=13)
    ax1.set_yticks(np.arange(0, y_max, y1_step))
    ax1.set_ylim(0,y_max)
    ax1.set_xlim(-0.001, spectrometer_resolutions[-1]+0.001)
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

    if type== "conc":
        # Create charts for dependence of error on resolution
        for data in range(0, len(datas)):
            ax2.plot(data_cnv[1:,0], abs(data_cnv[1:,data+1]-data_cnv[0,data+1])*100, 'k:', marker=markers[data], markersize=5, linewidth=1, label=datas[data].split(".")[0]) 
        for data in range(0, len(datas)):
            ax2.plot(data_simple_deconv[1:,0], abs(data_simple_deconv[1:,data+1]-data_simple_deconv[0,data+1])*100, 'r--', marker=markers[data], markersize=5, linewidth=1) 
        for data in range(0, len(datas)):
            ax2.plot(data_numeric_deconv[1:,0], abs(data_numeric_deconv[1:,data+1]-data_numeric_deconv[0,data+1])*100,  'b-', marker=markers[data], markersize=5, linewidth=1) 
    if type == "thickness":
        # Create charts for dependence of error on resolution
        for data in range(0, len(datas)):
            ax2.plot(data_cnv[1:,0], abs((data_cnv[1:,data+1]-data_cnv[0,data+1])/(2*data_cnv[0,data+1]))*100, 
                    'k-', marker=markers[data], markersize=5, linewidth=1, label=datas[data].split(".")[0]) 
        for data in range(0, len(datas)):
            ax2.plot(data_simple_deconv[1:,0], abs((data_simple_deconv[1:,data+1]-data_simple_deconv[0,data+1])/(2*data_simple_deconv[0,data+1]))*100,
                    'r-', marker=markers[data], markersize=5, linewidth=1) 
        for data in range(0, len(datas)):
            ax2.plot(data_numeric_deconv[1:,0], abs((data_numeric_deconv[1:,data+1]-data_numeric_deconv[0,data+1])/(2*data_numeric_deconv[0,data+1]))*100,
                    'b-', marker=markers[data], markersize=5, linewidth=1) 

    ax2.set_xlabel('relative resolution of spectrometer, ΔE/E', fontsize=13)
    ax2.set_xticks(np.arange(0, spectrometer_resolutions[-1]+0.001, 0.01))
    if type== "conc":
        ax2.set_ylabel('elemental composition error, %', fontsize=13)
    if type== "thickness":
        ax2.set_ylabel('film thickness error, %', fontsize=13)
    ax2.set_xlim(0, spectrometer_resolutions[-1]+0.001)
    ax2.set_ylim(0,error_max)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.minorticks_on()
    ax2.grid()
    ax2.text(.93, .98,"B", transform=ax2.transAxes, ha="center", va="top", size=15, weight='bold')

    plt.savefig(save_path+os.sep+type+"_"+calc_name+"_with_"+broadening_kernel_type+"_kernel"+logging_options+".png", dpi=300)
    plt.show()
    
    
def create_animated_chart(f, kernel_type, deltaEtoE, step,raw_en):
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot()
    energy_per_frame_scale = 20
    # do broadening convolution at different stages of k
    g = broadening_kernel(kernel_type, deltaEtoE)(raw_en,len(raw_en)*step)
    g = g[g>.001]
    Lg = len(g)
    Lf = len(f)
    en = np.arange(raw_en[0]*step, (len(raw_en)+len(g))*step, step)
    L = len(en)
    cnv_matrix = np.zeros((L, L))
    for k in range (1,L):
        for j in range (1,k): cnv_matrix[k][j]=cnv_matrix[k-1][j]
        g = broadening_kernel(kernel_type, deltaEtoE)(raw_en,en[k])
        for i in range(max(0, k + 1 - Lg), min(k+Lg, Lf)):
            cnv_matrix[k,k] += f[i] * g[i]
    cnv_matrix /= max(cnv_matrix[L-1][int(Emin/step):int(Emax/step)])
    
    def get_cnv_array(k):
        return cnv_matrix[k][0:k]    
        
    def animate(k):
        k=k*energy_per_frame_scale
        g = broadening_kernel(kernel_type, deltaEtoE)(en,en[k])
        cnv=get_cnv_array(k)

        ax.clear()
        plt.title('Demonstration of broadening convolution with dE/E='+str(deltaEtoE)+
                  '. Green line is proportional to the red area')
        
        plt.ylim(0,1)
        plt.xlim(0, en[-1]/1000)
        plt.xticks(np.arange(0, en[-1]/1000+1, 1))
        plt.minorticks_on()
        plt.xlabel('energy, keV', fontsize=13)
        plt.ylabel('intensity, a.u.', fontsize=13)
        
        en1=en/1000
        f2 = np.interp(en,raw_en, f)
        ax.plot(en1, f2,  'b', label='Initial signal')
        ax.plot(en1, g, 'r', label='Kernel')
        ax.plot(en1[0:k], cnv, 'c--', label='Convoluted', linewidth=3.0)
        fill_curve = np.minimum(f2, g)
        ax.fill_between(en1, np.zeros(len(en)), fill_curve, color='r', alpha=0.5, interpolate=True)
        plt.legend()   
        
    anim = animation.FuncAnimation(fig, animate, frames=range(1,int(raw_en[-1]/(step*energy_per_frame_scale))), 
                                   interval=2*(25000/(raw_en[-1])), repeat=False)
    anim.save('gifs'+os.sep+'animated_broad_conv_dEtoE='+str(deltaEtoE)+'_'+calc_name+'.gif')
    #fig.show()
    plt.show()