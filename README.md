# Green's Function approach to Temperature Response to CO2 

## Data storage

All data is stored at: '/net/fs11/d0/emfreese/CO2_GF/cmip6_data/' (utils has the path saved)

## Notebooks

**utils.py** has many of the functions and dictionaries that are used across notebooks. 

**1_CO2_GF_creation**: creates the Green's Functions, mostly using functions as defined in the utils.py. Run this first.

**2.1_evaluation_1pct_emis_profile**: creates the emissions profile for the 1pct CO2 runs to evaluate how well the Green's Functions represent the models

**2.2_evaluation_1pct_convolution**: convolves the 1pct CO2 emissions profiles with our Green's Function, and creates a dataset on the difference in temperature from the 1pct to the Control run

**2.3_evaluation_1pct_comparison**: compares the convolved 1pct CO2 output to the actual CMIP6 model 1pct output. This is for evaluating how well we do


**3.1_enroads_base_hist_ssp245_emis_profile**: creates the emissions profile for the historical + ssp245 CO2 runs. This solves the issue of providing us a temperature to build on for future temperature changes, as we can only vary CO2, but we need to account for other factors as well as historical emissions. Specific to the EnROADS approach.

**3.2_enroads_base_hist_ssp245_convolution**: convolves the emissions profile for the historical + ssp245 CO2 runs with our Green's Function and creates a dataset on the difference in temperature from the ssp245 (nat, all forcing, and GHG only forcing) and the Control. This solves the issue of providing us a temperature to build on for future temperature changes, as we can only vary CO2, but we need to account for other factors as well as historical emissions. Specific to the EnROADS approach.

**3.3_enroads_base_hist_ssp245_evaluation**: evaluates what the 'baseline' that we are creating looks like-- it is the subtraction of a convolution of the Green's function with the ssp245 CO2 emissions from the ssp245 temperature, providing an estimate of temperature change due to historical emissions/forcing from all factors + future temperature change due to everything except CO2. This solves the issue of providing us a temperature to build on for future temperature changes, as we can only vary CO2, but we need to account for other factors as well as historical emissions. Specific to the EnROADS approach. We can then convolve any EnROADS CO2 emissions scenario with the Green's function and add it onto this 'baseline'. 


