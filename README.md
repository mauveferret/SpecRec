# LEIS Energy Spectra Reconstruction/Recognition

Low-Energy Ion Scattering (LEIS) is a surface analysis technique that involves bombarding a sample with low-energy ions (typically 0.5-10 keV) and analyzing the energy spectrum of the scattered ions at specific angles. This technique is particularly sensitive to the outermost atomic layers of the sample.

![LEIS_scheme](https://github.com/mauveferret/SpecRec/blob/main/docs/pics/LEIS_Scheme.png?raw=true)

The **SpecRec** package is designed for comprehensive analysis of low-energy ion scattering spectroscopy data. It specifically addresses two main challenges: accounting for the transmission function of electrostatic analyzers and providing automated elemental analysis with surface concentration assessment for both simulated and experimental spectra. The package consists of two independent modules:

1. [`spectraConvDeconv_tools`](https://github.com/mauveferret/SpecRec/blob/main/tools/spectraConvDeconv_tools.py): Reconstructs the true energy distributions of ions that have been distorted by the electrostatic spectrometer
2. [`LEIS_tools`](https://github.com/mauveferret/SpecRec/blob/main/tools/LEIS_tools.py): Calculates elastic peak positions and sensitivity factors, while providing automated spectra analysis capabilities

The [`spectraConvDeconv_tools`](https://github.com/mauveferret/SpecRec/blob/main/tools/spectraConvDeconv_tools.py) module was developed as part of research published in Elsevier's journal:

[`N. Efimov, D. Sinelnikov, D. Kolodko, M. Grishaev, and I. Nikitin, 'On the reconstruction of LEIS spectra after distortion by an electrostatic energy analyzer', Applied Surface Science, vol. 676, p. 161006, Dec. 2024, doi.org/10.1016/j.apsusc.2024.161006`](https://doi.org/10.1016/j.apsusc.2024.161006)

Current development focuses on enhancing the [`LEIS_tools`](https://github.com/mauveferret/SpecRec/blob/main/tools/LEIS_tools.py) module for quantitative surface characterization using LEIS spectra. A scientific paper detailing these developments is in preparation.

## Description

### LEIS Tools

For detailed information about using LEIS tools, please refer to our comprehensive [tutorial file](https://github.com/mauveferret/SpecRec/blob/main/main_example.ipynb). This tutorial covers basic theory, practical examples, and implementation details.

### Energy Spectra Convolution/Deconvolution

The energy spectra module provides sophisticated tools for handling analyzer transmission effects. **SpecRec** offers advanced post-processing capabilities for experimental spectra measured by electrostatic or magnetic particle separators. Key features include:

- Spectrum smoothing and noise addition
- Convolution and deconvolution operations with multiple kernel shapes:
    - Gaussian
    - Triangle
    - Rectangle
    - Custom arbitrary shapes
- Support for both constant and broadening full width at half maximum (FWHM)

These capabilities make **SpecRec** particularly valuable for mass analysis and ion scattering spectroscopy applications.

The theoretical foundation and detailed methodology are thoroughly documented in our [recent publication](https://doi.org/10.1016/j.apsusc.2024.161006). **SpecRec** implements both analytical and numerical approaches to spectrum reconstruction:

1. Analytical methods developed by:
     - [Zhabrev and Zhdanov V.A.](https://inis.iaea.org/search/search.aspx?orig_q=RN:11571670)
     - [Urusov V.A.](http://link.springer.com/10.1134/S1063785010050196)

2. Modern numerical methods, including:
     - [TwoMey's approach](https://dl.acm.org/doi/10.1145/321150.321157)

These methods specifically address the Fredholm Integral Equation of the first kind, offering advantages over traditional convolution integral equations. As demonstrated by [Yu. K. Golikov et al.](https://cyberleninka.ru/article/n/ob-apparatnoy-funktsii-elektrostaticheskih-elektronnyh-spektrometrov), this approach is essential for accurate analysis of electrostatic and magnetic spectrometer data.

![Spectrum Reconstruction Example](https://github.com/mauveferret/SpecRec/blob/main/out/sim_Ne18keV32deg_HDW/spec_reconstr_sim_Ne18keV32deg_HDW_with_gauss_kernel.png?raw=true)

## Getting Started

### Installation and Dependencies

This program is entirely Python-based and requires Python to be installed on your system. If you don't have Python installed, you can download it from the official [Python website](https://www.python.org/downloads/). The program requires the following Python libraries:

* numpy - for numerical computations
* scipy - for scientific calculations
* inteq - for solving integral equations
* plotly - for interactive visualizations
* matplotlib - for static plotting
* brokenaxes - for complex axis manipulations

You can install all dependencies using:
```bash
pip install -r requirements.txt
```

If you experience issues with `pip`, please refer to the [official Python packaging guide](https://packaging.python.org/en/latest/tutorials/installing-packages/). Alternatively, you can use the `uv` package manager by running `uv run` in the working directory.

### Executing the Program

The recommended entry point is our comprehensive [tutorial notebook](https://github.com/mauveferret/SpecRec/blob/main/main_example.ipynb), which contains fundamental theory and practical examples of using this package.

While the program doesn't provide a graphical user interface, we recommend using [Visual Studio Code](https://code.visualstudio.com/), a free, cross-platform, and open-source IDE with excellent Python support. However, you can also run the scripts directly from the command line without an IDE.

### Project Structure

The project consists of several key components:

* `spectraConvDeconv_tools.py` - Core library containing all functions and global variables for spectral analysis. This module serves as a foundation for other components.

* `LEIS_tools.py` - Specialized module for analyzing ion energy losses, scattering angle deviations, cross-sections, and intensity correction factors.

* `general_plots.py` - Visualization tool for generating charts based on spectrometer transmission functions. Supports various kernel types ("gauss", "triangle", "rectangle", "LMM") and handles both simulated and experimental data.

#### Directory Structure:
* `/tools` - Contains module sources

* `/examples` - Contains representative LEIS case studies demonstrating spectrometer resolution effects and error analysis.

* `/raw_data` - Houses raw simulation and experimental data, as referenced in our published paper.

* `/out` - Contains all generated outputs, including figures and processed spectral data.

* `/out/gifs` - Stores animated visualizations of convolution processes with broadening kernels, generated via `general_plots.py`.

* `/docs` - Contain pictures for tutorials and some docs


### Recommendations

1. Please note that execution of example scripts may require significant processing time, especially for complex calculations.
2. For optimal performance, ensure your system meets the minimum requirements listed in the dependencies section.
3. When working with large datasets, consider using smaller test samples first.

## Help and Support

If you have questions about this program or encounter any execution problems, please reach out through:
- Telegram: Contact [mauveferret](https://t.me/mauveferret)
- Email: NEEfimov@mephi.ru
- GitHub Issues: Open an issue in the project repository

## License

This project is licensed under the GNU General Public License. For complete terms and conditions, please see the [LICENSE](https://github.com/mauveferret/SpecRec/blob/main/LICENSE.md) file in the repository.

## Acknowledgements

We extend our gratitude to:

* The [Inteq](https://github.com/mwt/inteq) project team for their Python library dedicated to solving Fredholm and Volterra equations using the [TwoMey](https://dl.acm.org/doi/10.1145/321150.321157) method, which has been instrumental in LEIS energy spectra reconstruction.
* Ivan Nikitin for developing [LEIS_calculator](https://elibrary.ru/item.asp?id=54049055), from which the Cross-section submodule of **SpecRec** was adapted.
* The scientific community for their valuable feedback and contributions.

