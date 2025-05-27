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
import matplotlib.ticker as ticker

from scipy.optimize import minimize

# changing working directoru to the SpecRec dir
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(parent_dir) 
BASE_LIB_PATH = "tools"
sys.path.insert(1, os.getcwd()+os.sep+BASE_LIB_PATH)
import LEIS_tools as leis
#import spectraConvDeconv_tools as SCD

# Presets
spectrum_path0 = os.getcwd()+os.sep+"raw_data"+os.sep+"exp_AuPd"
leis.Emin = 5000 # eV
leis.Emax = 15000 # eV
dE = 2 # eV
# SCD.step = dE # eV
filter_window = 100 # eV
# R - relative energy resolution of spectrometer
R = 0.01
plt.figure(figsize=(12, 8))

do_spectra_charts = True

# Load reference spectra
exp_spectra = os.listdir(spectrum_path0)
for spectrum in exp_spectra:
    if "ref_Ne_Au" in spectrum and not "ref_Ne_Au_late" in spectrum:
        ref_Ne_Au = leis.spectrum(spectrum_path0+os.sep+spectrum, filter_window, step=dE)
    if "ref_Ne_Pd" in spectrum  and not "ref_Ne_Pd_late"  in spectrum:
        ref_Ne_Pd = leis.spectrum(spectrum_path0+os.sep+spectrum, filter_window, step=dE)
    if "ref_Ne_Pd_late" in spectrum:
        ref_Ne_Pd_late = leis.spectrum(spectrum_path0+os.sep+spectrum, filter_window, step=dE)
    if "ref_Ne_Au_late" in spectrum:
        ref_Ne_Au_late = leis.spectrum(spectrum_path0+os.sep+spectrum, filter_window, step=dE)
    if "ref_Ar_Au" in spectrum:
        ref_Ar_Au = leis.spectrum(spectrum_path0+os.sep+spectrum, filter_window, step=dE)
    if "ref_Ar_Pd" in spectrum:
        ref_Ar_Pd = leis.spectrum(spectrum_path0+os.sep+spectrum, filter_window, step=dE)

def makeSpectrumByTwoRefs(E, Pd_coef, Au_coef):

    ll = [x*Pd_coef*ref_Ar_Pd.spectrum_max for x in np.interp(E, ref_Ar_Pd.spectrum_en, ref_Ar_Pd.spectrum_int)] 
    yy = [x*Au_coef*ref_Ar_Au.spectrum_max for x in np.interp(E, ref_Ar_Au.spectrum_en, ref_Ar_Au.spectrum_int)] 
    return [x + y for x, y in zip(ll, yy)]

def common_minimization_func(coeffs, data, ref_a, ref_b):
    a, b = coeffs
    S_combined = a * ref_a + b * ref_b
    return np.sum((data - S_combined) ** 2)

# Load experimental spectra and calculate the concentration of Au and Pd
i = 0
i_ar = 0
i_ne = 0
for spectrum in exp_spectra:
    if not "ref" in spectrum:
        data = leis.spectrum(spectrum_path0+os.sep+spectrum, filter_window, step=dE)
        if "Ne" in data.incident_atom:
            Pd_signal = leis.norm(data.spectrum_int) - np.interp(data.spectrum_en, ref_Ne_Au.spectrum_en, ref_Ne_Au.spectrum_int)
        elif "Ar" in data.incident_atom:
            Pd_signal = leis.norm(data.spectrum_int) - np.interp(data.spectrum_en, ref_Ar_Au.spectrum_en, ref_Ar_Au.spectrum_int)
        else:
            Pd_signal = leis.norm(data.spectrum_int)
            print(f"No reference was found for the {data.incident_atom} incident atom")
       # Calculate the concentration of Au and Pd based on the SemiRef approach and the sensitivity factors

        int_Pd = leis.peak(Pd_signal)/leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Pd")
        int_Au = 1/leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Au")
        conc_Au_semiRef_cross = int_Au/(int_Au+int_Pd)*100
        
        # Calculate the concentration of Au and Pd based on the Young's fitting model 
        young_fitting = leis.fitted_spectrum(data, "Pd", "Au")
        conc_Au_fitting = young_fitting.get_concentration()
        
        if "Ne" in data.incident_atom:
            Emax = 14800
            leis.Emax = Emax
            try:
                if True:
                 
                    Pd_spec = np.interp(data.spectrum_en[:int(Emax/leis.step)], ref_Ne_Pd.spectrum_en[:int(Emax/leis.step)], ref_Ne_Pd.spectrum_int[:int(Emax/leis.step)])*ref_Ne_Pd.spectrum_max
                    Au_spec = np.interp(data.spectrum_en[:int(Emax/leis.step)], ref_Ne_Au.spectrum_en[:int(Emax/leis.step)], ref_Ne_Au.spectrum_int[:int(Emax/leis.step)])*ref_Ne_Au.spectrum_max
                    result = minimize(common_minimization_func, [0.5, 0.5], args=(data.spectrum_int[:int(Emax/leis.step)]*data.spectrum_max, Pd_spec, Au_spec), method='Nelder-Mead')
                    Pd_coeff, Au_coeff = result.x
                    conc_Au_etalon =  Au_coeff*100   

                    #plt.figure(figsize=(12, 8))
                    #plt.plot(data.spectrum_en, data.spectrum_int*data.spectrum_max)
                    #plt.plot(ref_Ne_Pd.spectrum_en, ref_Ne_Pd.spectrum_int*ref_Ne_Pd.spectrum_max*Pd_coeff)
                    #plt.plot(ref_Ne_Au.spectrum_en, ref_Ne_Au.spectrum_int*ref_Ne_Au.spectrum_max*Au_coeff)
                    #plt.legend()
                    #plt.show()
                    #print(f"etalon 1 {Pd_coeff}   2   {Au_coeff}   =  {Au_coeff/(Pd_coeff+Au_coeff)}")
                    
                    # for semi-etalon method
                    Pd_spec = leis.norm(np.interp(data.spectrum_en[:int(Emax/leis.step)], ref_Ne_Pd.spectrum_en[:int(Emax/leis.step)], ref_Ne_Pd.spectrum_int[:int(Emax/leis.step)]))
                    Au_spec = leis.norm(np.interp(data.spectrum_en[:int(Emax/leis.step)], ref_Ne_Au.spectrum_en[:int(Emax/leis.step)], ref_Ne_Au.spectrum_int[:int(Emax/leis.step)]))
                    result = minimize(common_minimization_func, [0.5, 0.5], args=(leis.norm(data.spectrum_int[:int(Emax/leis.step)]), Pd_spec, Au_spec), method='Nelder-Mead')
                    Pd_coeff, Au_coeff = result.x          
                    Pd_coeff = Pd_coeff / leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Pd")
                    Au_coeff = Au_coeff / leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Au")
                    #print(f"semi-etalon 1 {Pd_coeff}   2   {Au_coeff}   =  {Au_coeff/(Pd_coeff+Au_coeff)}")
                    #conc_Au_semiRef_cross =  Au_coeff/(Pd_coeff+Au_coeff)*100     
                else:
                    #conc_Au_etalon = data.spectrum_max/ref_Ne_Au_late.spectrum_max*100  #young_fitting.get_concentration()
                    Pd_spec = np.interp(data.spectrum_en[:int(Emax/leis.step)], ref_Ne_Pd_late.spectrum_en[:int(Emax/leis.step)], ref_Ne_Pd_late.spectrum_int[:int(Emax/leis.step)])*ref_Ne_Pd_late.spectrum_max
                    Au_spec = np.interp(data.spectrum_en[:int(Emax/leis.step)], ref_Ne_Au_late.spectrum_en[:int(Emax/leis.step)], ref_Ne_Au_late.spectrum_int[:int(Emax/leis.step)])*ref_Ne_Au_late.spectrum_max
                    result = minimize(common_minimization_func, [0.5, 0.5], args=(data.spectrum_int[:int(Emax/leis.step)]*data.spectrum_max, Pd_spec, Au_spec), method='Nelder-Mead')
                    Pd_coeff, Au_coeff = result.x
                    conc_Au_etalon =  Au_coeff*100   
                    
                    # for semi-etalon method
                    Pd_spec = leis.norm(np.interp(data.spectrum_en[:int(Emax/leis.step)], ref_Ne_Pd_late.spectrum_en[:int(Emax/leis.step)], ref_Ne_Pd_late.spectrum_int[:int(Emax/leis.step)]))
                    Au_spec = leis.norm(np.interp(data.spectrum_en[:int(Emax/leis.step)], ref_Ne_Au_late.spectrum_en[:int(Emax/leis.step)], ref_Ne_Au_late.spectrum_int[:int(Emax/leis.step)]))
                    result = minimize(common_minimization_func, [0.5, 0.5], args=(leis.norm(data.spectrum_int[:int(Emax/leis.step)]), Pd_spec, Au_spec), method='Nelder-Mead')
                    Pd_coeff, Au_coeff = result.x                
                    Pd_coeff = Pd_coeff / leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Pd")
                    Au_coeff = Au_coeff / leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Au")
                    #print(f"semi-etalon 1 {Pd_coeff}   2   {Au_coeff}   =  {Au_coeff/(Pd_coeff+Au_coeff)}")
                    conc_Au_semiRef_cross =  Au_coeff/(Pd_coeff+Au_coeff)*100                      
                    
            except Exception as e:
                print(e)
                #conc_Au_etalon = data.spectrum_max/ref_Ne_Au.spectrum_max*100  #young_fitting.get_concentration()
        elif "Ar" in data.incident_atom:
                #   for etalon method
                
                Emax = 14400
                Pd_spec = np.interp(data.spectrum_en[:int(Emax/leis.step)], ref_Ar_Pd.spectrum_en[:int(Emax/leis.step)], ref_Ar_Pd.spectrum_int[:int(Emax/leis.step)])*ref_Ar_Pd.spectrum_max
                Au_spec = np.interp(data.spectrum_en[:int(Emax/leis.step)], ref_Ar_Au.spectrum_en[:int(Emax/leis.step)], ref_Ar_Au.spectrum_int[:int(Emax/leis.step)])*ref_Ar_Au.spectrum_max
                result = minimize(common_minimization_func, [0.5, 0.5], args=(data.spectrum_int[:int(Emax/leis.step)]*data.spectrum_max, Pd_spec, Au_spec), method='Nelder-Mead')
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
                Pd_spec = np.interp(data.spectrum_en[:int(Emax/leis.step)], ref_Ar_Pd.spectrum_en[:int(Emax/leis.step)], ref_Ar_Pd.spectrum_int[:int(Emax/leis.step)])
                Au_spec = np.interp(data.spectrum_en[:int(Emax/leis.step)], ref_Ar_Au.spectrum_en[:int(Emax/leis.step)], ref_Ar_Au.spectrum_int[:int(Emax/leis.step)])
                result = minimize(common_minimization_func, [0.5, 0.5], args=(data.spectrum_int[:int(Emax/leis.step)], Pd_spec, Au_spec), method='Nelder-Mead')
   
                # Извлечение оптимальных коэффициентов
                Pd_coeff, Au_coeff = result.x               
                Pd_coeff = Pd_coeff / leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Pd")
                Au_coeff = Au_coeff / leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Au")
                
                #print(f"semi-etalon 1 {Pd_coeff}   2   {Au_coeff}   =  {Au_coeff/(Pd_coeff+Au_coeff)}")
                conc_Au_semiRef_cross =  Au_coeff/(Pd_coeff+Au_coeff)*100           
                
        print(f"{data.calc_name[0:16]} {data.incident_atom} {conc_Au_semiRef_cross:.2f} % {conc_Au_etalon:.2f} % {conc_Au_fitting:.2f} %")
        
        if do_spectra_charts:
            plt.figure(figsize=(12, 8))
            plt.plot(data.spectrum_en/1000, leis.norm(data.spectrum_int), "k-", label="Экспериментальный спектр Au50Pd50", linewidth=3, alpha=0.9)
            box  = f"Концентрация золота = {conc_Au_semiRef_cross:.2f} ат. %"   
            if "Ne" in data.incident_atom:
                plt.plot(ref_Ne_Au.spectrum_en/1000, ref_Ne_Au.spectrum_int, "r--"  ,label="Полуэталонный Au", linewidth=3, alpha=0.8)
                plt.plot(ref_Ne_Pd.spectrum_en/1000-0.1, ref_Ne_Pd.spectrum_int*max(Pd_signal), "b--", label="Полуэталонный Pd", linewidth=3, alpha=0.9)
                plt.plot(data.spectrum_en[:int(14500/dE)]/1000, Pd_signal[:int(14500/dE)], "g-", label="Сигнал Pd (= чёрный - красный)", linewidth=3, alpha=0.7)
                plt.xlim(12,15)
                plt.text(12.2, 0.7, box, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

            else:
                plt.plot(ref_Ar_Au.spectrum_en/1000, ref_Ar_Au.spectrum_int, "r--"  ,label="Полуэталонный Au", linewidth=3, alpha=0.8)
                plt.plot(ref_Ar_Pd.spectrum_en/1000, ref_Ar_Pd.spectrum_int*max(Pd_signal),  "b--", label="Полуэталонный Pd", linewidth=3, alpha=0.9)
                plt.plot(data.spectrum_en[:int(14300/dE)]/1000, Pd_signal[:int(14300/dE)], "g-", label="Сигнал Pd (= чёрный - красный)", linewidth=3, alpha=0.7)
                plt.xlim(10,15)
                plt.text(10.2, 0.7, box, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

            #plt.plot(data.spectrum_en, young_fitting.get_fitted_spectrum(), label="Аппроксимация по формуле Йанга")
            #plt.plot(data.spectrum_en, young_fitting.get_elastic_part("Au"), "--", label="Упругая часть Au по формуле Йанга")
            #plt.plot(data.spectrum_en, young_fitting.get_inelastic_part("Au"), "--", label="Неупругая часть Au по формуле Йанга")     
            #plt.plot(data.spectrum_en, young_fitting.get_elastic_part("Pd"), "--", label="Упругая часть Pd по формуле Йанга")
            #plt.plot(data.spectrum_en, young_fitting.get_inelastic_part("Pd"), "--", label="Неупругая часть Pd по формуле Йанга")      
            plt.ylim(0, 1)
            #plt.xlim(12,15)
            plt.xlabel('энергия, кэВ', fontsize=16)
            plt.ylabel('интенсивность, норм.', fontsize=16)
            plt.title(f"Экспериментальный спектр {spectrum}", y=1.05) 
            plt.legend(fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.minorticks_on()
            plt.show()
        else:
            if "Ne" in data.incident_atom:
                i_ne+=1        
                plt.plot(i_ne, conc_Au_semiRef_cross, "x", color="red", markersize=20)
                #plt.annotate(spectrum.split("Ne")[0].split("2024")[1], (i_ne,conc_Au_semiRef_cross + 0.2))
                #plt.plot(i_ne, conc_Au_etalon, "o", color="red" )
                #plt.annotate(spectrum.split("Ne")[0].split("2025")[1], (i_ne,conc_Au_etalon + 0.2))
                #plt.plot(i_ne, conc_Au_fitting, "*", color="red")
                #plt.annotate(spectrum.split("Ne")[0].split("2025")[1], (i_ne,conc_Au_fitting + 0.2))

            else:
                i_ar+=1
                plt.plot(i_ar, conc_Au_semiRef_cross, "*", color= "green", markersize=20)
                #plt.annotate(spectrum.split("Ar")[0].split("2024")[1], (i_ar,conc_Au_semiRef_cross + 0.2))
                #plt.plot(i_ar, conc_Au_etalon, "o", color="green")
                #plt.annotate(spectrum.split("Ar")[0].split("2025")[1], (i_ar,conc_Au_etalon + 0.2))
                #plt.plot(i_ar, conc_Au_fitting, "*", color="green")
                #plt.annotate(spectrum.split("Ar")[0].split("2025")[1], (i_ar,conc_Au_fitting + 0.2))
        # Store concentrations for statistics
        if i == 0:
            Ne_conc = []
            Ar_conc = []
        
        if data.incident_atom == "Ne":
            Ne_conc.append(conc_Au_semiRef_cross)
        else:
            Ar_conc.append(conc_Au_semiRef_cross)

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
if not do_spectra_charts:
    plt.plot(-1, 0, "*", color= "green", label ="Аргон 15 кэВ")
    plt.plot(-1, 0, "x", color="red", label ="Неон 15 кэВ")
    plt.axhline(y=50, color='black', linestyle=':', alpha=0.7, linewidth=2)
    plt.xlim(left=0)
    plt.ylim(30, 70)
    plt.xlabel('номер спектра', fontsize = 15)
    plt.ylabel('концентрация Au, %', fontsize = 15)
    plt.title('Concentration of Au in the Au50Pd50 samples for experimental LEIS spectra. Ne - RED, Ar - GREEN', y=1.05)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.minorticks_on()    
    plt.legend( fontsize = 15)
    plt.show()
