"""
This program allows to provide quantification of the element analysis by LEIS

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

If you have questions regarding this program, please contact NEEfimov@mephi.ru
"""

import os,  sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize

# changing working directoru to the SpecRec dir
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(parent_dir) 
BASE_LIB_PATH = "tools"
sys.path.insert(1, os.getcwd()+os.sep+BASE_LIB_PATH)
import LEIS_tools as leis
import spectraConvDeconv_tools as SCD

dE = 2 # eV
leis.step = dE # eV
SCD.step = dE 

##############################            PRESETS          ##########################################################

spectrum_path0 = os.getcwd()+os.sep+"raw_data"+os.sep+"exp_AuPd"
#spectrum_path0 = "D:\Спектры\\250603 Ne 15 keV AuPd"
#spectrum_path0 = "D:\Спектры\\250604 Ne 15 keV AuPd heating"
#spectrum_path0 = "D:\Спектры\\250605 Ar Ne 15 keV AuPd heating"
#spectrum_path0 = "O:\OneDrive\Проекты\Крокодил\Данные\Спектры\\250604 Ne 15 keV AuPd heating"
#spectrum_path0 = "O:\OneDrive\Проекты\Крокодил\Данные\Спектры\\250605 Ar Ne 15 keV AuPd heating"
leis.Emin = 7000 # eV
leis.Emax = 15000 # eV

# smoothing parameter
filter_window = 80 # eV

# R - relative energy resolution of spectrometer
R = 0.01
 
do_spectra_charts = True

####################################################################################################################

#plt.figure(figsize=(12, 8))

# Load reference spectra
exp_spectra = os.listdir(spectrum_path0)

for spectrum in exp_spectra:
    if "ref_Ne_Au" in spectrum and not "ref_Ne_Au_late" in spectrum:
        ref_Ne_Au = leis.spectrum(spectrum_path0+os.sep+spectrum, filter_window, step=dE)
        ref_Ne_Au.shift_spectrum_en(20)
    if "ref_Ne_Pd" in spectrum  and not "ref_Ne_Pd_late"  in spectrum:
        ref_Ne_Pd = leis.spectrum(spectrum_path0+os.sep+spectrum, filter_window, step=dE)
        ref_Ne_Pd.shift_spectrum_en(110)
    if "ref_Ne_Pd_late" in spectrum:
        ref_Ne_Pd_late = leis.spectrum(spectrum_path0+os.sep+spectrum, filter_window, step=dE)
    if "ref_Ne_Au_late" in spectrum:
        ref_Ne_Au_late = leis.spectrum(spectrum_path0+os.sep+spectrum, filter_window, step=dE)
    if "ref_Ar_Au" in spectrum:
        ref_Ar_Au = leis.spectrum(spectrum_path0+os.sep+spectrum, filter_window, step=dE)
        ref_Ar_Au.shift_spectrum_en(-10)
    if "ref_Ar_Pd" in spectrum:
        ref_Ar_Pd = leis.spectrum(spectrum_path0+os.sep+spectrum, filter_window, step=dE)
        ref_Ar_Au.shift_spectrum_en(20)
    if "ref_Kr_Au" in spectrum:
        ref_Kr_Au = leis.spectrum(spectrum_path0+os.sep+spectrum, filter_window, step=dE)
        #ref_Kr_Au.shift_spectrum_en(0)
    if "ref_Kr_Pd" in spectrum:
        ref_Kr_Pd = leis.spectrum(spectrum_path0+os.sep+spectrum, filter_window, step=dE)
        #ref_Ar_Au.shift_spectrum_en(20)

# for SemiEmpirical Fitting
def makeSpectrumByTwoRefs(E, Pd_coef, Au_coef):
    ll = [x*Pd_coef*ref_Ar_Pd.spectrum_max for x in np.interp(E, ref_Ar_Pd.spectrum_en, ref_Ar_Pd.spectrum_int)] 
    yy = [x*Au_coef*ref_Ar_Au.spectrum_max for x in np.interp(E, ref_Ar_Au.spectrum_en, ref_Ar_Au.spectrum_int)] 
    return ll+yy #[x + y for x, y in zip(ll, yy)]

# for deconvolution of spectrum for several elements
def common_minimization_func(coeffs, data, ref_a, ref_b):
    a, b = coeffs
    S_combined = a * ref_a + b * ref_b
    return np.sum((data - S_combined) ** 2)

# Load experimental spectra and calculate the concentration of Au and Pd
i = 0
i_ar = 0
i_ne = 0
i_kr = 0
#plt.figure(figsize=(12, 8))

#plt.figure(figsize=(12, 8))

for spectrum in exp_spectra:
    if not "ref" in spectrum and ".txt" in spectrum :
            
        data = leis.spectrum(spectrum_path0+os.sep+spectrum, -1, step=dE)
        # Basic semietalon method with one reference
        if "Ne" in data.incident_atom:
            Pd_signal = leis.norm(data.spectrum_int) - leis.norm(np.interp(data.spectrum_en, ref_Ne_Au.spectrum_en, ref_Ne_Au.spectrum_int))
        elif "Ar" in data.incident_atom:
            Pd_signal = leis.norm(data.spectrum_int) - leis.norm(np.interp(data.spectrum_en, ref_Ar_Au.spectrum_en, ref_Ar_Au.spectrum_int))
        elif "Kr" in data.incident_atom:
            Pd_signal = leis.norm(data.spectrum_int) - leis.norm(np.interp(data.spectrum_en, ref_Kr_Au.spectrum_en, ref_Kr_Au.spectrum_int))
        else:
            Pd_signal = leis.norm(data.spectrum_int)
            print(f"WARNING: No reference was found for the {data.incident_atom} incident atom")
            
       # Calculate the concentration of Au and Pd based on the SemiRef approach and the sensitivity factors
        int_Pd = leis.peak(Pd_signal)/leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Pd")
        int_Au = 1/leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Au")
        
        conc_Au_semiRef_cross = int_Au/(int_Au+int_Pd)*100
        Au_rel_int = data.spectrum_max/1E-8*100
        Pd_rel_int = leis.peak(Pd_signal)*data.spectrum_max/1E-8*100
       
        # Calculate the concentration of Au and Pd based on the Young's fitting model 
        #young_fitting = leis.fitted_spectrum(data, "Pd", "Au")
        #conc_Au_fitting = young_fitting.get_concentration()

        if "Ne" in data.incident_atom:
            Emin = leis.Emin
            Emax = 14800
            leis.Emax = Emax
            try:
                if True:
                    
                    # ETALON method
                    Pd_spec = np.interp(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ne_Pd.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ne_Pd.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)])*ref_Ne_Pd.spectrum_max
                    Au_spec = np.interp(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ne_Au.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ne_Au.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)])*ref_Ne_Au.spectrum_max
                    result = minimize(common_minimization_func, [0.5, 0.5], args=(data.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)]*data.spectrum_max, Pd_spec, Au_spec), method='Nelder-Mead')
                    Pd_coeff, Au_coeff = result.x
                    
                    conc_Au_etalon =  Au_coeff*100   
                    #conc_Au_etalon = data.spectrum_max/ref_Ne_Au.spectrum_max*100*56/41
                    
                    #plt.figure(figsize=(12, 8))
                    #plt.plot(data.spectrum_en, data.spectrum_int*data.spectrum_max)
                    #plt.plot(ref_Ne_Pd.spectrum_en, ref_Ne_Pd.spectrum_int*ref_Ne_Pd.spectrum_max*Pd_coeff)
                    #plt.plot(ref_Ne_Au.spectrum_en, ref_Ne_Au.spectrum_int*ref_Ne_Au.spectrum_max*Au_coeff)
                    #plt.legend()
                    #plt.show()
                    
                    # for SEMI-ETALON method
                    Pd_spec = (np.interp(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ne_Pd.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ne_Pd.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)]))
                    Pd_spec = Pd_spec/max(Pd_spec)
                    Au_spec = (np.interp(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ne_Au.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ne_Au.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)]))
                    Au_spec = Au_spec/max(Au_spec)
                    result = minimize(common_minimization_func, [0.5, 0.5], args=((data.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)]), Pd_spec, Au_spec), method='Nelder-Mead')
                    Pd_coeff, Au_coeff = result.x          
    
                    Au_Deconv_spectrum = Au_spec*Au_coeff
                    Pd_Deconv_spectrum = Pd_spec*Pd_coeff
                    #print(f"semi-etalon 1 {Pd_coeff}   2   {Au_coeff}   =  {Au_coeff/(Pd_coeff+Au_coeff)}")
                    
                    Pd_coeff_cross = Pd_coeff / leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Pd")
                    Au_coeff_cross = Au_coeff / leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Au")
                    
                    conc_Au_semiRef_cross_Deconvolution =  Au_coeff_cross/(Pd_coeff_cross+Au_coeff_cross)*100                     
                    
            except Exception as e:
                print(e)
                #conc_Au_etalon = data.spectrum_max/ref_Ne_Au.spectrum_max*100  #young_fitting.get_concentration()
        elif "Ar" in data.incident_atom:
                #   for etalon method
                Emin = 13300
                Emin = leis.Emin
                Emax = 14400
                Pd_spec = np.interp(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ar_Pd.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ar_Pd.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)])*ref_Ar_Pd.spectrum_max
                Au_spec = np.interp(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ar_Au.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ar_Au.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)])*ref_Ar_Au.spectrum_max
                result = minimize(common_minimization_func, [0.5, 0.5], args=(data.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)]*data.spectrum_max, Pd_spec, Au_spec), method='Nelder-Mead')
                Pd_coeff, Au_coeff = result.x
                
                #plt.figure(figsize=(12, 8))
                #plt.plot(data.spectrum_en, data.spectrum_int*data.spectrum_max)
                #plt.plot(ref_Ar_Pd.spectrum_en, ref_Ar_Pd.spectrum_int*ref_Ar_Pd.spectrum_max)
                #plt.plot(ref_Ar_Au.spectrum_en, ref_Ar_Au.spectrum_int*ref_Ar_Au.spectrum_max)
                #plt.plot(data.spectrum_en, makeSpectrumByTwoRefs(data.spectrum_en, Pd_coeff, Au_coeff), label="fitted")
                #plt.legend()
                #plt.show()
                #print(f"etalon 1 {Pd_coeff}   2   {Au_coeff}   =  {Au_coeff/(Pd_coeff+Au_coeff)}")
                conc_Au_etalon =  Au_coeff*100   
                
                # for semi-etalon method
                Pd_spec = (np.interp(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ar_Pd.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ar_Pd.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)]))
                Pd_spec = Pd_spec/max(Pd_spec)
                Au_spec = (np.interp(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ar_Au.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ar_Au.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)]))
                Au_spec = Au_spec/max(Au_spec)
                result = minimize(common_minimization_func, [0.5, 0.5], args=(data.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)], Pd_spec, Au_spec), method='Nelder-Mead')
   
                # Извлечение оптимальных коэффициентов
                Pd_coeff, Au_coeff = result.x               
    
                Au_Deconv_spectrum = Au_spec*Au_coeff
                Pd_Deconv_spectrum = Pd_spec*Pd_coeff
                
                Pd_coeff_cross = Pd_coeff / leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Pd")
                Au_coeff_cross = Au_coeff / leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Au")
                
                conc_Au_semiRef_cross_Deconvolution =  Au_coeff_cross/(Pd_coeff_cross+Au_coeff_cross)*100             
        elif "Kr" in data.incident_atom:
                #   for etalon method
                Emin = 7000
                Emin = leis.Emin
                Emax = 11000
                Pd_spec = np.interp(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Kr_Pd.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Kr_Pd.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)])*ref_Kr_Pd.spectrum_max
                Au_spec = np.interp(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Kr_Au.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Kr_Au.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)])*ref_Kr_Au.spectrum_max
                result = minimize(common_minimization_func, [0.5, 0.5], args=(data.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)]*data.spectrum_max, Pd_spec, Au_spec), method='Nelder-Mead')
                Pd_coeff, Au_coeff = result.x
                
                #plt.figure(figsize=(12, 8))
                #plt.plot(data.spectrum_en, data.spectrum_int*data.spectrum_max)
                #plt.plot(ref_Ar_Pd.spectrum_en, ref_Ar_Pd.spectrum_int*ref_Ar_Pd.spectrum_max)
                #plt.plot(ref_Ar_Au.spectrum_en, ref_Ar_Au.spectrum_int*ref_Ar_Au.spectrum_max)
                #plt.plot(data.spectrum_en, makeSpectrumByTwoRefs(data.spectrum_en, Pd_coeff, Au_coeff), label="fitted")
                #plt.legend()
                #plt.show()
                #print(f"etalon 1 {Pd_coeff}   2   {Au_coeff}   =  {Au_coeff/(Pd_coeff+Au_coeff)}")
                conc_Au_etalon =  Au_coeff*100   
                
                # for semi-etalon method
                Pd_spec = (np.interp(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Kr_Pd.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Kr_Pd.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)]))
                Pd_spec = Pd_spec/max(Pd_spec)
                Au_spec = (np.interp(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Kr_Au.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Kr_Au.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)]))
                Au_spec = Au_spec/max(Au_spec)
                result = minimize(common_minimization_func, [0.5, 0.5], args=(data.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)], Pd_spec, Au_spec), method='Nelder-Mead')
   
                # Извлечение оптимальных коэффициентов
                Pd_coeff, Au_coeff = result.x               
    
                Au_Deconv_spectrum = Au_spec*Au_coeff
                Pd_Deconv_spectrum = Pd_spec*Pd_coeff

                Pd_coeff_cross = Pd_coeff / leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Pd")
                Au_coeff_cross = Au_coeff / leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Au")
                
                conc_Au_semiRef_cross_Deconvolution =  Au_coeff_cross/(Pd_coeff_cross+Au_coeff_cross)*100                   
            
        #print(f"{data.calc_name[0:16]} {data.incident_atom} {conc_Au_semiRef_cross:.2f} % {conc_Au_etalon:.2f} % {conc_Au_fitting:.2f} %")

        if do_spectra_charts and "Ne" in data.incident_atom:
            plt.figure(figsize=(12, 8))
            plt.plot(data.spectrum_en/1000, leis.norm(data.spectrum_int), "-", color="grey", linewidth=3, alpha=0.5)
            plt.plot(data.spectrum_en/1000, scipy.signal.savgol_filter(leis.norm(data.spectrum_int), int(300/leis.step), 5), "k-",  label="Экспериментальный спектр Au-Pd", linewidth=3)

            box  = f"Концентрация золота {conc_Au_semiRef_cross_Deconvolution:.1f} ат. % \n"   
            if "Ne" in data.incident_atom:
                plt.plot(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)]/1000, Au_Deconv_spectrum, "r--"  ,label="Полуэталонный Au", linewidth=3, alpha=0.8)
                plt.plot(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)]/1000, Pd_Deconv_spectrum, "b--", label="Полуэталонный Pd", linewidth=3, alpha=0.9)
                #plt.plot(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)]/1000, Au_Deconv_spectrum+Pd_Deconv_spectrum, "m-.", label="Сумма", linewidth=3, alpha=0.7)
                plt.plot(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)]/1000, scipy.signal.savgol_filter(leis.norm(data.spectrum_int)[int(Emin/leis.step):int(Emax/leis.step)], int(300/leis.step), 5) - Au_Deconv_spectrum, "g-", label="Сигнал Pd (= чёрный - красный)", linewidth=3, alpha=0.7)

                #plt.plot(data.spectrum_en[:int(14400/dE)]/1000, scipy.signal.savgol_filter(Pd_signal[:int(14400/dE)], int(300/leis.step), 5), "g-", label="Сигнал Pd (= чёрный - красный)", linewidth=3, alpha=0.7)
                plt.xlim(7,15)
                plt.text(7.05, 0.7, box, fontsize=14)

            elif "Ar":
                plt.plot(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)]/1000, Au_Deconv_spectrum, "r--"  ,label="Полуэталонный Au", linewidth=3, alpha=0.8)
                plt.plot(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)]/1000, Pd_Deconv_spectrum, "b--", label="Полуэталонный Pd", linewidth=3, alpha=0.9)
                plt.plot(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)]/1000, Au_Deconv_spectrum+Pd_Deconv_spectrum, "m-.", label="Сумма", linewidth=3, alpha=0.7)
                plt.plot(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)]/1000, scipy.signal.savgol_filter(leis.norm(data.spectrum_int)[int(Emin/leis.step):int(Emax/leis.step)], int(300/leis.step), 5) - Au_Deconv_spectrum, "g-", label="Сигнал Pd (= чёрный - красный)", linewidth=3, alpha=0.7)

                #plt.plot(data.spectrum_en[:int(14100/dE)]/1000, scipy.signal.savgol_filter(Pd_signal[:int(14100/dE)], int(300/leis.step), 5), "g-", label="Сигнал Pd (= чёрный - красный)", linewidth=3, alpha=0.7)

                plt.xlim(7,15)
                plt.text(7.2, 0.7, box, fontsize=14)

            elif "Kr":
                plt.plot(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)]/1000, Au_Deconv_spectrum, "r--"  ,label="Полуэталонный Au", linewidth=3, alpha=0.8)
                plt.plot(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)]/1000, Pd_Deconv_spectrum, "b--", label="Полуэталонный Pd", linewidth=3, alpha=0.9)
                plt.plot(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)]/1000, Au_Deconv_spectrum+Pd_Deconv_spectrum, "m-.", label="Сумма", linewidth=3, alpha=0.7)
                plt.plot(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)]/1000, scipy.signal.savgol_filter(leis.norm(data.spectrum_int)[int(Emin/leis.step):int(Emax/leis.step)], int(300/leis.step), 5) - Au_Deconv_spectrum, "g-", label="Сигнал Pd (= чёрный - красный)", linewidth=3, alpha=0.7)

                #plt.plot(data.spectrum_en[:int(14100/dE)]/1000, scipy.signal.savgol_filter(Pd_signal[:int(14100/dE)], int(300/leis.step), 5), "g-", label="Сигнал Pd (= чёрный - красный)", linewidth=3, alpha=0.7)

                plt.xlim(5,11)
                plt.text(5, 0.7, box, fontsize=14)
            #plt.plot(data.spectrum_en, young_fitting.get_fitted_spectrum(), label="Аппроксимация по формуле Йанга")
            #plt.plot(data.spectrum_en, young_fitting.get_elastic_part("Au"), "--", label="Упругая часть Au по формуле Йанга")
            #plt.plot(data.spectrum_en, young_fitting.get_inelastic_part("Au"), "--", label="Неупругая часть Au по формуле Йанга")     
            #plt.plot(data.spectrum_en, young_fitting.get_elastic_part("Pd"), "--", label="Упругая часть Pd по формуле Йанга")
            #plt.plot(data.spectrum_en, young_fitting.get_inelastic_part("Pd"), "--", label="Неупругая часть Pd по формуле Йанга")      
            
            plt.ylim(0.03, 1)
            plt.xlabel('энергия, кэВ', fontsize=20)
            plt.ylabel('интенсивность, норм.', fontsize=20)
            plt.title(f"Экспериментальный спектр {spectrum}", y=1.05) 
            plt.legend(fontsize=18)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.minorticks_on()
            plt.show()
        else:
            if "Ne" in data.incident_atom:
                i_ne+=1        
                #plt.plot(i_ne, conc_Au_semiRef_cross, "o", color="red", markersize=12)
                plt.plot(i_ne, conc_Au_semiRef_cross_Deconvolution, "x", color="red", markersize=17)

                #plt.plot(i_ne, conc_Au_semiRef_cross_Deconvolution, "s", color="orange", markersize=10, alpha=0.7)

                #plt.annotate(f"{t}", (i_ne,conc_Au_semiRef_cross + 0.2))
                
                #plt.plot(i_ne, Au_rel_int, "o", color="blue", markersize=15)
                #plt.plot(i_ne, Pd_rel_int, "o", color="black", markersize=15)

                #plt.plot(i_ne, conc_Au_etalon, "o", color="red" )
                #plt.annotate(f"{t}", (i_ne,conc_Au_etalon + 0.2))
                #plt.plot(i_ne, conc_Au_fitting, "*", color="red")
                #plt.annotate(spectrum.split("Ne")[0].split("2025")[1], (i_ne,conc_Au_fitting + 0.2))

            elif "Ar" in data.incident_atom:
                i_ar+=1
                plt.plot(i_ar, conc_Au_semiRef_cross_Deconvolution, "*", color= "green", markersize=20)
                #plt.plot(i_ar, conc_Au_semiRef_cross_Deconvolution, "s", color="olive", markersize=10, alpha=0.7)

                #plt.annotate(spectrum.split("Ar")[0].split("2025")[1], (i_ar,conc_Au_semiRef_cross + 0.2))
                
                #plt.plot(i_ar, conc_Au_etalon, "^", color="green")
                #plt.annotate(spectrum.split("Ar")[0].split("2025")[1], (i_ar,conc_Au_etalon + 0.2))
                
                #plt.plot(i_ar, conc_Au_fitting, "*", color="green")
                #plt.annotate(spectrum.split("Ar")[0].split("2025")[1], (i_ar,conc_Au_fitting + 0.2))
                
                #plt.plot(i_ar, Au_rel_int, "<", color="blue", markersize=15)
                #plt.plot(i_ar, Pd_rel_int, "<", color="black", markersize=15)
            else:
                i_kr+=1
                plt.plot(i_kr, conc_Au_semiRef_cross_Deconvolution, "o", color= "blue", markersize=17)
                
        # Store concentrations for statistics
        if i == 0:
            Ne_conc = []
            Ar_conc = []
            Kr_conc = []

        if data.incident_atom == "Ne":
            Ne_conc.append(conc_Au_semiRef_cross_Deconvolution)
        elif data.incident_atom == "Ar":
            Ar_conc.append(conc_Au_semiRef_cross_Deconvolution)
        elif data.incident_atom == "Kr":
            Kr_conc.append(conc_Au_semiRef_cross_Deconvolution)

        i+=1

        # After last spectrum, print statistics
        if i == len([s for s in exp_spectra if "ref" not in s]):
            if Ne_conc:
                print(f"\nNe incident atoms:")
                print(f"Average Au concentration: {np.mean(Ne_conc):.2f}%")
                print(f"Standard deviation: {np.std(Ne_conc):.2f}%")
            if Ar_conc:
                print(f"\nAr incident atoms:")
                print(f"Average Au concentration: {np.mean(Ar_conc):.2f}%")
                print(f"Standard deviation: {np.std(Ar_conc):.2f}%")
            if Kr_conc:
                print(f"\nKr incident atoms:")
                print(f"Average Au concentration: {np.mean(Kr_conc):.2f}%")
                print(f"Standard deviation: {np.std(Kr_conc):.2f}%")
if not do_spectra_charts:
    #plt.plot(-1, 0, "*", color= "green", label ="Аргон 15 кэВ")
    plt.plot(-1, 0, "x", color="red", label ="Ne 15 кэВ \"Полуэталонный\"")
    #plt.plot(-1, 0, "o", color="red", label ="Ne 15 кэВ \"Эталонный\" норм.")
    plt.plot(-1, 0, "*", color="green", label ="Ar 15 кэВ \"Полуэталонный\"")
    plt.plot(-1, 0, "o", color="blue", label ="Kr 11 кэВ \"Полуэталонный\"")

    #plt.plot(-1, 0, "o", color="blue", label ="пик золота / 1E-8 * 100%")
    #plt.plot(-1, 0, "<", color="black", label ="пик палладия / 1E-8 * 100%")

    plt.axhline(y=50, color='black', linestyle=':', alpha=0.8, linewidth=4)
    plt.xlim(left=0)
    plt.ylim(30, 80)
    plt.xlabel('номер спектра', fontsize = 24)
    plt.ylabel('концентрация Au, %', fontsize = 24)
    #plt.title(f'Concentration of Au in the Au50Pd50 samples for experimental LEIS spectra. \n Sample Temperature is shown in Annotations \n {spectrum_path0}', y=1.02)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.minorticks_on()    
    plt.legend( fontsize = 20)
    plt.show()
