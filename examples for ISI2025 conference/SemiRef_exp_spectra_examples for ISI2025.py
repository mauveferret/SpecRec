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

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize

# Language selection - choose between "rus" and "eng"
charts_lang = "eng"  # Change to "rus" for Russian or "eng" for English

# Translation dictionaries
TRANSLATIONS = {
    "rus": {
        "title_experimental": "Экспериментальный спектр",
        "title_concentration": "Концентрация Au в образцах Au50Pd50",
        "label_energy": "энергия, кэВ",
        "label_intensity": "интенсивность, норм.",
        "label_concentration": "концентрация Au, %",
        "label_spectrum_number": "номер спектра",
        "legend_experimental": "Экспериментальный спектр Au-Pd",
        "legend_au": "Полуэталонный Au",
        "legend_pd": "Полуэталонный Pd",
        "legend_sum": "Сумма",
        "legend_pd_signal": "Сигнал Pd (= чёрный - красный)",
        "legend_ne": "Ne 15 кэВ \"Полуэталонный\"",
        "legend_ar": "Ar 15 кэВ \"Полуэталонный\"",
        "legend_kr": "Kr 11 кэВ \"Полуэталонный\"",
        "concentration_box": "Концентрация золота {:.1f} ат. % \n",
        "stats_ne": "Атомы Ne:",
        "stats_ar": "Атомы Ar:",
        "stats_kr": "Атомы Kr:",
        "stats_avg": "Средняя концентрация Au: {:.2f}%",
        "stats_std": "Стандартное отклонение: {:.2f}%",
        "xps" : "результаты РФЭС (51.95 ат. %)",
        "expected" : "ожидаемая",
        "mean" : "в среднем",
        "at" : "ат.",
        "warning_no_reference": "WARNING: No reference was found for the {} incident atom",
        "beam" : ""
    },
    "eng": {
        "title_experimental": "Experimental spectrum",
        "title_concentration": "Concentration of Au in Au50Pd50 samples",
        "label_energy": "energy, keV",
        "label_intensity": "intensity, norm.",
        "label_concentration": "Au concentration, %",
        "label_spectrum_number": "spectrum number",
        "legend_experimental": "Experimental Au-Pd spectrum",
        "legend_au": "Semi-reference Au",
        "legend_pd": "Semi-reference Pd",
        "legend_sum": "Fitting (= blue + red)",
        "legend_pd_signal": "Pd signal (= black - red)",
        "legend_ne": "Ne 15 keV ",
        "legend_ar": "Ar 15 keV ",
        "legend_kr": "Kr 11 keV ",
        "concentration_box": "Gold concentration {:.1f} at. % \n",
        "stats_ne": "Ne incident atoms:",
        "stats_ar": "Ar incident atoms:",
        "stats_kr": "Kr incident atoms:",
        "stats_avg": "Average Au concentration: {:.2f}%",
        "stats_std": "Standard deviation: {:.2f}%",
        "warning_no_reference": "WARNING: No reference was found for the {} incident atom",
        "xps" : "XPS data (51.95 at. %)",
        "expected" : "expected",
        "mean" : "mean",
        "at" : "at.",
        "beam" : "beam"
    }
}

def t(key):
    """Helper function to get translation for current language"""
    return TRANSLATIONS[charts_lang][key]

# changing working directory to the SpecRec dir
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
leis.Emin = 7000 # eV
leis.Emax = 15000 # eV

# smoothing parameter
filter_window = 120 # eV

# R - relative energy resolution of spectrometer
R = 0.01
 
do_spectra_charts = True
plot_ions = "Ne"
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
        ref_Ne_Pd.shift_spectrum_en(90)
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
    if "ref_Kr_Pd" in spectrum:
        ref_Kr_Pd = leis.spectrum(spectrum_path0+os.sep+spectrum, filter_window, step=dE)

# for SemiEmpirical Fitting
def makeSpectrumByTwoRefs(E, Pd_coef, Au_coef):
    ll = [x*Pd_coef*ref_Ar_Pd.spectrum_max for x in np.interp(E, ref_Ar_Pd.spectrum_en, ref_Ar_Pd.spectrum_int)] 
    yy = [x*Au_coef*ref_Ar_Au.spectrum_max for x in np.interp(E, ref_Ar_Au.spectrum_en, ref_Ar_Au.spectrum_int)] 
    return ll+yy

# for deconvolution of spectrum for several elements
def common_minimization_func(coeffs, data, ref_a, ref_b):
    a, b = coeffs
    S_combined = a * ref_a + b * ref_b
    return np.sum((data - S_combined) ** 2)

# Load experimental spectra and calculate the concentration of Au and Pd
i = 0
#i_ar = 0
#i_ne = 0
#i_kr = 0

ne_data = []    # Будет хранить кортежи (номер_файла, концентрация)
ar_data = []    # Будет хранить кортежи (номер_файла, концентрация)
kr_data = []

for spectrum in exp_spectra:
    if not "ref" in spectrum and ".txt" in spectrum :
            
        data = leis.spectrum(spectrum_path0+os.sep+spectrum, -1, step=dE)
        # Basic semireference method with one reference
        if "Ne" in data.incident_atom:
            Pd_signal = leis.norm(data.spectrum_int) - leis.norm(np.interp(data.spectrum_en, ref_Ne_Au.spectrum_en, ref_Ne_Au.spectrum_int))
        elif "Ar" in data.incident_atom:
            Pd_signal = leis.norm(data.spectrum_int) - leis.norm(np.interp(data.spectrum_en, ref_Ar_Au.spectrum_en, ref_Ar_Au.spectrum_int))
        elif "Kr" in data.incident_atom:
            Pd_signal = leis.norm(data.spectrum_int) - leis.norm(np.interp(data.spectrum_en, ref_Kr_Au.spectrum_en, ref_Kr_Au.spectrum_int))
        else:
            Pd_signal = leis.norm(data.spectrum_int)
            print(t("warning_no_reference").format(data.incident_atom))
            
       # Calculate the concentration of Au and Pd based on the SemiRef approach and the sensitivity factors
        int_Pd = leis.peak(Pd_signal)/leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Pd")
        int_Au = 1/leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Au")
        
        conc_Au_semiRef_cross = int_Au/(int_Au+int_Pd)*100
        Au_rel_int = data.spectrum_max/1E-8*100
        Pd_rel_int = leis.peak(Pd_signal)*data.spectrum_max/1E-8*100
       
        if "Ne" in data.incident_atom:
            Emin = 13900 #leis.Emin
            Emax = 14800
            leis.Emax = Emax
            try:
                if True:
                    # reference method
                    Pd_spec = np.interp(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ne_Pd.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ne_Pd.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)])*ref_Ne_Pd.spectrum_max
                    Au_spec = np.interp(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ne_Au.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ne_Au.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)])*ref_Ne_Au.spectrum_max
                    result = minimize(common_minimization_func, [0.5, 0.5], args=(data.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)]*data.spectrum_max, Pd_spec, Au_spec), method='Nelder-Mead')
                    Pd_coeff, Au_coeff = result.x
                    
                    conc_Au_reference = Au_coeff*100   
                    
                    # for SEMI-reference method
                    Pd_spec = (np.interp(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ne_Pd.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ne_Pd.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)]))
                    Pd_spec = Pd_spec/max(Pd_spec)
                    Au_spec = (np.interp(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ne_Au.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ne_Au.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)]))
                    Au_spec = Au_spec/max(Au_spec)
                    result = minimize(common_minimization_func, [0.5, 0.5], args=((data.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)]), Pd_spec, Au_spec), method='Nelder-Mead')
                    Pd_coeff, Au_coeff = result.x          
    
                    Au_Deconv_spectrum = Au_spec*Au_coeff
                    Pd_Deconv_spectrum = Pd_spec*Pd_coeff
                    
                    Pd_coeff_cross = Pd_coeff / leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Pd")
                    Au_coeff_cross = Au_coeff / leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Au")
                    
                    conc_Au_semiRef_cross_Deconvolution = Au_coeff_cross/(Pd_coeff_cross+Au_coeff_cross)*100                     
                    
            except Exception as e:
                print(e)
        elif "Ar" in data.incident_atom:
            Emin = 12500 #leis.Emin
            Emax = 14400
            Pd_spec = np.interp(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ar_Pd.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ar_Pd.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)])*ref_Ar_Pd.spectrum_max
            Au_spec = np.interp(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ar_Au.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ar_Au.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)])*ref_Ar_Au.spectrum_max
            result = minimize(common_minimization_func, [0.5, 0.5], args=(data.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)]*data.spectrum_max, Pd_spec, Au_spec), method='Nelder-Mead')
            Pd_coeff, Au_coeff = result.x
                
            conc_Au_reference = Au_coeff*100   
            
            # for semi-reference method
            Pd_spec = (np.interp(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ar_Pd.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ar_Pd.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)]))
            Pd_spec = Pd_spec/max(Pd_spec)
            Au_spec = (np.interp(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ar_Au.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Ar_Au.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)]))
            Au_spec = Au_spec/max(Au_spec)
            result = minimize(common_minimization_func, [0.5, 0.5], args=(data.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)], Pd_spec, Au_spec), method='Nelder-Mead')
            Pd_coeff, Au_coeff = result.x               
    
            Au_Deconv_spectrum = Au_spec*Au_coeff
            Pd_Deconv_spectrum = Pd_spec*Pd_coeff
            
            Pd_coeff_cross = Pd_coeff / leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Pd")
            Au_coeff_cross = Au_coeff / leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Au")
            
            conc_Au_semiRef_cross_Deconvolution = Au_coeff_cross/(Pd_coeff_cross+Au_coeff_cross)*100             
        elif "Kr" in data.incident_atom:
            Emin = 7000
            Emax = 11000
            Pd_spec = np.interp(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Kr_Pd.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Kr_Pd.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)])*ref_Kr_Pd.spectrum_max
            Au_spec = np.interp(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Kr_Au.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Kr_Au.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)])*ref_Kr_Au.spectrum_max
            result = minimize(common_minimization_func, [0.5, 0.5], args=(data.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)]*data.spectrum_max, Pd_spec, Au_spec), method='Nelder-Mead')
            Pd_coeff, Au_coeff = result.x
                
            conc_Au_reference = Au_coeff*100   
            
            # for semi-reference method
            Pd_spec = (np.interp(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Kr_Pd.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Kr_Pd.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)]))
            Pd_spec = Pd_spec/max(Pd_spec)
            Au_spec = (np.interp(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Kr_Au.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)], ref_Kr_Au.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)]))
            Au_spec = Au_spec/max(Au_spec)
            result = minimize(common_minimization_func, [0.5, 0.5], args=(data.spectrum_int[int(Emin/leis.step):int(Emax/leis.step)], Pd_spec, Au_spec), method='Nelder-Mead')
            Pd_coeff, Au_coeff = result.x               
    
            Au_Deconv_spectrum = Au_spec*Au_coeff
            Pd_Deconv_spectrum = Pd_spec*Pd_coeff

            Pd_coeff_cross = Pd_coeff / leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Pd")
            Au_coeff_cross = Au_coeff / leis.get_cross_section(data.incident_atom, data.E0, data.scattering_angle, "Au")
            
            conc_Au_semiRef_cross_Deconvolution = Au_coeff_cross/(Pd_coeff_cross+Au_coeff_cross)*100                   
            
        if do_spectra_charts and plot_ions in data.incident_atom:
            plt.figure(figsize=(12, 8))
            plt.plot(data.spectrum_en/1000, leis.norm(data.spectrum_int), "-", color="grey", linewidth=3, alpha=0.5)
            plt.plot(data.spectrum_en/1000, scipy.signal.savgol_filter(leis.norm(data.spectrum_int), int(300/leis.step), 5), "k-", label=t("legend_experimental"), linewidth=3)

            box = t("concentration_box").format(conc_Au_semiRef_cross_Deconvolution)
            
            
            angle = 32
            if "Ne" in data.incident_atom:
                O_peak_local = leis.get_energy_by_angle(data.E0, leis.get_mass_by_element("O")/leis.get_mass_by_element(data.incident_atom), angle)
                plt.vlines(x=O_peak_local/1000, ymin=0, ymax=data.spectrum_int[int(O_peak_local/leis.step)], linewidth = 3, colors='black', linestyles='dotted', alpha = 0.6)
                plt.text(O_peak_local/1000, data.spectrum_int[int(O_peak_local/leis.step)] + 0.01, 'O', ha='center', va='bottom', fontsize = 18, color = 'black' )
            
            if "Ne" in data.incident_atom:
                angle = 33
            else:
                angle = 31
            Au_peak_local = leis.get_energy_by_angle(data.E0, leis.get_mass_by_element("Au")/leis.get_mass_by_element(data.incident_atom), angle)
            
            if not "Kr" in data.incident_atom:
                plt.vlines(x=Au_peak_local/1000, ymin=0, ymax=data.spectrum_int[int(Au_peak_local/leis.step)], linewidth = 3, colors='r', linestyles='dotted', alpha = 0.6)
                plt.text(Au_peak_local/1000, data.spectrum_int[int(Au_peak_local/leis.step)] + 0.01, 'Au', ha='center', va='bottom', fontsize = 18, color = 'r' )
                Pd_peak_local = leis.get_energy_by_angle(data.E0, leis.get_mass_by_element("Pd")/leis.get_mass_by_element(data.incident_atom), angle)
                plt.vlines(x=Pd_peak_local/1000, ymin=0, ymax=data.spectrum_int[int(Pd_peak_local/leis.step)], linewidth =3 , colors='b', linestyles='dotted', alpha = 0.6)
                plt.text(Pd_peak_local/1000, data.spectrum_int[int(Pd_peak_local/leis.step)] + 0.01, 'Pd', ha='center', va='bottom', fontsize = 18, color = 'b' )   
            else:
                plt.vlines(x=Au_peak_local/1000, ymin=0, ymax=data.spectrum_int[int(Au_peak_local/leis.step)]+0.2, linewidth = 3, colors='r', linestyles='dotted', alpha = 0.6)
                plt.text(Au_peak_local/1000, data.spectrum_int[int(Au_peak_local/leis.step)] + 0.01+0.2, 'Au', ha='center', va='bottom', fontsize = 18, color = 'r' )
                Pd_peak_local = leis.get_energy_by_angle(data.E0, leis.get_mass_by_element("Pd")/leis.get_mass_by_element(data.incident_atom), angle)
                plt.vlines(x=Pd_peak_local/1000, ymin=0, ymax=data.spectrum_int[int(Pd_peak_local/leis.step)]+0.2, linewidth =3 , colors='b', linestyles='dotted', alpha = 0.6)
                plt.text(Pd_peak_local/1000, data.spectrum_int[int(Pd_peak_local/leis.step)] + 0.01+0.2, 'Pd', ha='center', va='bottom', fontsize = 18, color = 'b' )

            if "Ne" in data.incident_atom:
                plt.plot(ref_Ne_Au.spectrum_en/1000, leis.norm(ref_Ne_Au.spectrum_int)*max(Au_Deconv_spectrum), "r--", label=t("legend_au"), linewidth=3, alpha=0.8)
                plt.plot(ref_Ne_Pd.spectrum_en/1000, leis.norm(ref_Ne_Pd.spectrum_int)*max(Pd_Deconv_spectrum), "b--", label=t("legend_pd"), linewidth=3, alpha=0.8)
                #plt.plot(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)]/1000, scipy.signal.savgol_filter(leis.norm(data.spectrum_int)[int(Emin/leis.step):int(Emax/leis.step)], int(300/leis.step), 5) - Au_Deconv_spectrum, "g-", label=t("legend_pd_signal"), linewidth=3, alpha=0.7)
                plt.plot(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)]/1000, Au_Deconv_spectrum+Pd_Deconv_spectrum, "g-.", label=t("legend_sum"), linewidth=4, alpha=0.8)

                plt.xlim(9,15)
                plt.text(9.2, 0.67, box, fontsize=14)

            elif "Ar" in data.incident_atom:

                plt.plot(ref_Ar_Au.spectrum_en/1000, leis.norm(ref_Ar_Au.spectrum_int)*max(Au_Deconv_spectrum), "r--", label=t("legend_au"), linewidth=3, alpha=0.8)
                plt.plot(ref_Ar_Pd.spectrum_en/1000, leis.norm(ref_Ar_Pd.spectrum_int)*max(Pd_Deconv_spectrum), "b--", label=t("legend_pd"), linewidth=3, alpha=0.9)
                plt.plot(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)]/1000, Au_Deconv_spectrum+Pd_Deconv_spectrum, "g-.", label=t("legend_sum"), linewidth=5, alpha=0.8)
                #plt.plot(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)]/1000, scipy.signal.savgol_filter(leis.norm(data.spectrum_int)[int(Emin/leis.step):int(Emax/leis.step)], int(300/leis.step), 5) - Au_Deconv_spectrum, "g-", label=t("legend_pd_signal"), linewidth=3, alpha=0.7)
                plt.xlim(9,15)
                plt.text(9.2, 0.65, box, fontsize=14)

            elif "Kr" in data.incident_atom:
                plt.plot(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)]/1000, Au_Deconv_spectrum, "r--", label=t("legend_au"), linewidth=3, alpha=0.8)
                plt.plot(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)]/1000, Pd_Deconv_spectrum, "b--", label=t("legend_pd"), linewidth=3, alpha=0.9)
                plt.plot(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)]/1000, Au_Deconv_spectrum+Pd_Deconv_spectrum, "g-.", label=t("legend_sum"), linewidth=4, alpha=0.8)
                #plt.plot(data.spectrum_en[int(Emin/leis.step):int(Emax/leis.step)]/1000, scipy.signal.savgol_filter(leis.norm(data.spectrum_int)[int(Emin/leis.step):int(Emax/leis.step)], int(300/leis.step), 5) - Au_Deconv_spectrum, "g-", label=t("legend_pd_signal"), linewidth=3, alpha=0.7)
                plt.xlim(5,11)
                plt.text(5.1, 0.65, box, fontsize=14)
            
            plt.ylim(0.02, 1)
            plt.xlabel(t("label_energy"), fontsize=20)
            plt.ylabel(t("label_intensity"), fontsize=20)
            plt.legend(loc="lower left", frameon=False, fontsize=18)
            #plt.title(f"{t('title_experimental')} {spectrum}", y=1.05) 
            plt.legend(fontsize=18, frameon=False)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            ax = plt.subplot(111)

            # Remove the top spine
            ax.spines['top'].set_visible(False)

            # Optionally, remove the right spine as well for a cleaner look
            ax.spines['right'].set_visible(False)
            plt.minorticks_on()
            plt.show()
        else:
            if "Ne" in data.incident_atom:
                ne_data.append(conc_Au_semiRef_cross_Deconvolution)
            elif "Ar" in data.incident_atom:
                ar_data.append(conc_Au_semiRef_cross_Deconvolution)
            elif "Kr" in data.incident_atom:
                kr_data.append(conc_Au_semiRef_cross_Deconvolution)

                
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

        i += 1

        # After last spectrum, print statistics
        if i == len([s for s in exp_spectra if "ref" not in s]):
            if Ne_conc:
                print(f"\n{t('stats_ne')}")
                print(t('stats_avg').format(np.mean(Ne_conc)))
                print(t('stats_std').format(np.std(Ne_conc)))
            if Ar_conc:
                print(f"\n{t('stats_ar')}")
                print(t('stats_avg').format(np.mean(Ar_conc)))
                print(t('stats_std').format(np.std(Ar_conc)))
            if Kr_conc:
                print(f"\n{t('stats_kr')}")
                print(t('stats_avg').format(np.mean(Kr_conc)))
                print(t('stats_std').format(np.std(Kr_conc)))

if not do_spectra_charts:
    plt.figure(figsize=(14, 10))
    all_data = []
    
    for conc in ne_data:
        all_data.append((0, conc, "red", "x", t("legend_ne")))
    
    for conc in ar_data:
        all_data.append((1, conc, "green", "*", t("legend_ar")))
    
    for conc in kr_data:
        all_data.append((2, conc, "blue", "o", t("legend_kr")))
    
    np.random.shuffle(all_data)
    x_positions = {0: 1, 1: 1, 2: 1}  
    for group, conc, color, marker, label in all_data:
        x = x_positions[group]
        plt.plot(x, conc, marker, color=color, markersize=16 if marker != "*" else 18)
        x_positions[group] += 1
    
    if ne_data:
        mean_ne = np.mean(ne_data)
        plt.axhline(y=mean_ne, color='red', linestyle='--', alpha=0.6, linewidth=2)
        plt.text(12.2, mean_ne+0.2, "$Ne^+$ "+t("beam"), fontsize=18, color="red")

    
    if ar_data:
        mean_ar = np.mean(ar_data)
        plt.axhline(y=mean_ar, color='green', linestyle='--', alpha=0.6, linewidth=2)
        plt.text(12.2, mean_ar+0.2, "$Ar^+$ "+t("beam"), fontsize=18, color="green")

    if kr_data:
        mean_kr = np.mean(kr_data)
        plt.axhline(y=mean_kr, color='blue', linestyle='--', alpha=0.6, linewidth=2) #label=f'Kr avg: {mean_kr:.1f}%'
        plt.text(12.2, mean_kr+0.2, "$Kr^+$ "+t("beam"), fontsize=18, color="blue")

    plt.axhline(y=50, color='black', linestyle='-', alpha=0.6, linewidth=4, label=t("expected"))
    plt.axhline(y=51.95, color='black', linestyle=':', alpha=0.6, linewidth=4, label=t("xps"))
    
    plt.fill_betweenx(np.linspace (np.mean(Ne_conc)-np.std(Ne_conc), np.mean(Ne_conc)+np.std(Ne_conc) ),0, 14, color='red', alpha=0.1)
    plt.fill_betweenx(np.linspace (np.mean(Ar_conc)-np.std(Ar_conc), np.mean(Ar_conc)+np.std(Ar_conc) ),0, 14, color='green', alpha=0.1)
    plt.fill_betweenx(np.linspace (np.mean(Kr_conc)-np.std(Kr_conc), np.mean(Kr_conc)+np.std(Kr_conc) ),0, 14, color='blue', alpha=0.1)

    plt.fill_betweenx(np.linspace (50, 51.95 ),0, 14, color='grey', alpha=0.6)
    
    
    # Добавляем легенду и оформление графика
    plt.plot(-1, 0, "o", color="blue", label=t("legend_kr")+f" ({t('mean')}={round(np.mean(Kr_conc))} {t('at')} %, σ={round(np.std(Kr_conc))} {t('at')} %)")
    plt.plot(-1, 0, "x", color="red", label=t("legend_ne")+f" ({t('mean')}={round(np.mean(Ne_conc))} {t('at')} %, σ={round(np.std(Ne_conc))} {t('at')} %)")
    plt.plot(-1, 0, "*", color="green", label=t("legend_ar")+f" ({t('mean')}={round(np.mean(Ar_conc))} {t('at')} %, σ={round(np.std(Ar_conc))} {t('at')} %)")
    
    plt.xlim(0, max(len(ne_data), len(ar_data), len(kr_data)) + 1)
    plt.ylim(30, 80)
    #plt.xlabel(t("label_spectrum_number"), fontsize=24)
    
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    plt.ylabel(t("label_concentration"), fontsize=24)
    plt.title(t("title_concentration"), y=1.02)
    plt.grid(True)
    plt.legend(loc="upper left", frameon=False, fontsize=18)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.minorticks_on()    
    plt.show()