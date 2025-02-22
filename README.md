# LEIS Energy Spectra Reconstruction

The program **SpecRec** is intended for the analysis of low-energy ion scattering spectroscopy data. In particular, it considers the transmission function, provides automated elemental analysis, and surface concentration estimations.

Its LEIS Spectra Convolution/Deconvolution module was written as part of a scientific paper published in Elsevier's journal:

[`N. Efimov, D. Sinelnikov, D. Kolodko, M. Grishaev, and I. Nikitin, ‘On the reconstruction of LEIS spectra after distortion by an electrostatic energy analyzer’, Applied Surface Science, vol. 676, p. 161006, Dec. 2024, doi.org/10.1016/j.apsusc.2024.161006`](https://doi.org/10.1016/j.apsusc.2024.161006)

Current work on the project involves writing a module for quantitative characterization of the surface using LEIS spectra and publishing a scientific paper on this topic.

![graphical_abstract](https://ars.els-cdn.com/content/image/1-s2.0-S0169433224017197-ga1.jpg)

## Description

\>>>>>>>>>>> **BRIEFLY**: just look at this [tutorial file](https://github.com/mauveferret/SpecRec/blob/main/main_example.ipynb)  <<<<<<<<<<<<

**SpecRec** can be used to post-process experimental spectra measured by electrostatic or magnetic separators of charged particles. It allows for smoothing, adding noise to the original spectra, and convolution and deconvolution with different kernel shapes (Gaussian, Triangle, Rectangle, and arbitrary) with constant or broadening full width at half maximum (FWHM). It can thus be useful in mass analysis and ion scattering spectroscopy.

A more detailed description of the algorithms and methods used is given in the [paper](https://doi.org/10.1016/j.apsusc.2024.161006). Briefly, **SpecRec** allows the use of analytical methods of spectra reconstruction suggested by [Zhabrev and Zhdanov V.A.](https://inis.iaea.org/search/search.aspx?orig_q=RN:11571670) and by [Urusov V.A.](http://link.springer.com/10.1134/S1063785010050196). It also allows the utilization of modern numerical methods, *inter alia* proposed by [TwoMey](https://dl.acm.org/doi/10.1145/321150.321157). Both groups of methods are intended for solving the Fredholm Integral Equation of the first kind, while most methods for spectra reconstruction are valid for classic convolution integral equations. Meanwhile, as shown by [Zhabrev and Zhdanov V.A.](https://inis.iaea.org/search/search.aspx?orig_q=RN:11571670) and by [Yu. K. Golikov et al.](https://cyberleninka.ru/article/n/ob-apparatnoy-funktsii-elektrostaticheskih-elektronnyh-spektrometrov), the use of classical convolution equations is not correct for most electrostatic and magnetic spectrometers, and the Fredholm or Volterra Integral Equations have to be used instead.

![header](https://github.com/mauveferret/SpecRec/blob/main/out/sim_Ne18keV32deg_HDW/spec_reconstr_sim_Ne18keV32deg_HDW_with_gauss_kernel.png?raw=true)

## Getting Started

### Installation and Dependencies

This is a fully Python-based program, so you need Python installed on your PC. If you do not have Python installed, you can use the following [link](https://www.python.org/downloads/). This program also requires the following Python libraries to be installed:

* numpy 
* scipy
* inteq 
* plotly
* matplotlib
* brokenaxes

You can install them all with the `pip install -r requirements.txt` command. If you have problems using `pip`, please read [this](https://packaging.python.org/en/latest/tutorials/installing-packages/). It was made with `uv` package manager, so you also can use it to get all dependencies instead of pip. To do it, just run `uv run` in the working directory  

### Executing the Program

AS the entrypoint a ipynb [tutorial file](https://github.com/mauveferret/SpecRec/blob/main/main_example.ipynb) is highly recommended. It contains some basic theory and practical examples with the use of this package. 

As the program does not provide a graphical user interface, the best way to launch it is with a text editor that has a built-in Python application launcher. A free, cross-platform, and open-source [Visual Studio Code](https://code.visualstudio.com/) is highly recommended. However, it's not strictly necessary, as the script can be launched as an executable without any IDE via the shell.

### File Structure (the information is little bit out of date, look example file)

* `spectraConvDeconv_tools.py` contains all functions and global variables for the study and serves as a library for other Python executables. It doesn't do anything by itself.

* `LEIS_tools.py` contains methods for determining ion energy losses, scattering angle deviations, cross-sections, intensity correction factors, etc.

* `general_plots.py` allows generating charts for a single specific transmission function of the spectrometer, which is given via resolution (dE/E) and kernel type ("gauss", "triangle", "rectangle", "LMM"). If the simulated charts are chosen, the program would first provide the convolution of the raw data and then deconvolute it by two methods (analytical and numerical). If experimental data is chosen, it would first be deconvoluted by two methods and then both results would be convoluted.

* Files in `examples` are examples for different representative LEIS cases, which allow creating dependencies of some qualitative estimations on the resolution of the spectrometer of a specific kernel as well as the dependencies of errors.

* The `raw_data` directory contains raw data from computer simulations and experiments. The source of the data is described in the paper mentioned at the beginning of this README.

* The `out` directory contains outputs of all Python scripts, including figures and data files with spectra.

* The `out/gifs` directory contains animated charts that show the process of convolution with a broadening kernel. These charts can be created with the `general_plots.py` script.

### Recommendations

1. Be patient, as `ex....` scripts require a lot of time!

## Help

If you have questions regarding this program or any execution problems, please contact [mauveferret](https://t.me/mauveferret) or NEEfimov@mephi.ru.

## License

This project is licensed under the GNU License - see the [this](https://github.com/mauveferret/SpecRec/blob/main/LICENSE.md)
 file for details.

## Acknowledgements

* [Inteq](https://github.com/mwt/inteq) for the Python library intended for Fredholm and Volterra equation solving with the [TwoMey](https://dl.acm.org/doi/10.1145/321150.321157) method. LEIS energy spectra reconstruction
* Ivan Nikitin for his [LEIS_calculator](https://elibrary.ru/item.asp?id=54049055), from which Cross-section submodule of **SpecRec** was taken.
