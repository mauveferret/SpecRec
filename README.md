# LEIS energy spectra reconstruction

The program **SpecRec** is intended as supporting material to the scientific paper published in the Elsevier's journal "Applied Surface Science": `You will see the reference here when the article would be published`

## Description

**SpecRec** can be used to postprocess experimental spectra measured by electrostatic or magnetic separators of charged particles. It allows to provide smoothing, noising of the original spectra, it convolution and deconvolution with different kernel's shapes (Gaussian, Triangle, Rectangle and arbitrary) with constant or broadening full width at half maximum (FWHM). It thus can be useful in mass analysis and ion scattering spectroscopy. 

More detailed description of the used algorithms and methods is given in `You will see the reference here when the article would be published`. Briefly, **SpecRec** allows to use analytical methods of spectra reconstruction suggested by [Zhabrev and Zhdanov V.A.](https://inis.iaea.org/search/search.aspx?orig_q=RN:11571670) and by [Urusov V.A.](http://link.springer.com/10.1134/S1063785010050196). Also it allows to utilize modern numerical methods, *inter alia* proposed by [TwoMey](https://dl.acm.org/doi/10.1145/321150.321157).  Both group of methods are intended for solving Fredholm Integral Equation of the first kind, while most methods for spectra reconstruction are valid for classic convolution integral equations.  Meanwhile, as was shown by [Zhabrev and Zhdanov V.A.](https://inis.iaea.org/search/search.aspx?orig_q=RN:11571670) and by [Yu. K. Golikov et. al.](https://cyberleninka.ru/article/n/ob-apparatnoy-funktsii-elektrostaticheskih-elektronnyh-spektrometrov), the use of classical convolution equations is not correct for the most of electrostatic and magnetic spectrometers and the Fredholm or Volterra Integral Equations has to be used instead.

![header](https://github.com/mauveferret/SpecRec/blob/main\out\sim_Ne18keV32deg_HDW\spec_reconstr_sim_Ne18keV32deg_HDW_with_gauss_kernel?raw=true)

## Getting Started

### Installation and Dependencies

This is a fully Python program, so you need it to be installed on your PC. If you do not have Python installed, you can use the following [link](https://www.python.org/downloads/). This program also requires the following python libraries to be installed:

* numpy 
* scipy
* inteq 
* plotly
* matplotlib

You can install it all with `pip install -r requirements.txt` command.  If you have problems with using `pip` please read [this](https://packaging.python.org/en/latest/tutorials/installing-packages/).


### Executing program

As soon as  the program does not provide graphical user interface, the best way to launch it is with some tex editor, which has a built-in python application launcher. A free, cross-platform and open-source [Visual Studio Code](https://code.visualstudio.com/) is highly recommended. However, it's not strictly necessary, as the script can be launched as an executable without any IDE via shell.

### File structure

* `spectraConvDeconvLib.py` contains all functions and global variables for the study and severs as library for another py executables. It doesn't do anything by itself.

* `general_plots.py` allows to generate charts for single specific transmission function of the spectrometer, which is given via resolution (dE/E) and kernel type("gauss", "triangle", "rectangle", "LMM"). If the simulated charts are chosen, the program would firstly provide the convolution of the raw data, and then deconvolute it by two methods (analytical and numerical). If experimental data is chosen, it would be firstly deconvoluted by two methods and then both results would be convoluted.

* files `ex...` are the examples for different representative LEIS cases, which alloes to create dependencies of some qualitative estimations on the resolution of the spectrometer of specific kernel as well as the dependencies of errors.

* dir `raw_data` indeed contains raw data of computer simulations and experiments. The sorce of the data is described in paper, mentioned in the beginning of this readme.

* dir `out` contain outputs of all python scripts, including figures and data files with spectra

* dir gifs contain animated charts, that shows a process of convolution with broadening kernel. This charts can be created with `general_plots.py` script.

### Recommendations

1. Be patient, as `ex....` scripts require a lot of time!

## Help

If you have questions regarding this program or any execution problems, please contact 
[mauveferret](https://t.me/mauveferret) or NEEfimov@mephi.ru

## License

This project is licensed under the GNU License - see the LICENSE.md file for details

## Acknowledgements


* [Inteq](https://github.com/mwt/inteq) for python lib, intended for Fredholm and Volterra equation solving with [TwoMey](https://dl.acm.org/doi/10.1145/321150.321157) method.
