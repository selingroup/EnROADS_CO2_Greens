import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cftime
import dask
import xarrayutils
import cartopy.crs as ccrs
from xmip.preprocessing import combined_preprocessing
from xmip.preprocessing import replace_x_y_nominal_lat_lon
from xmip.drift_removal import replace_time
from xmip.postprocessing import concat_experiments
import xmip.drift_removal as xm_dr
import xmip as xm
import xesmf as xe
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import cf_xarray as cfxr


########################### CMIP 6 DATA AND REGRIDDING ###################################

path_to_cmip6_data = '/net/fs11/d0/emfreese/CO2_GF/cmip6_data/'


#### initial import and data merging ####
#subroutines to import

def _import_combine_pulse_control(control_path, pulse_path, m):
    '''Import the pulse run and control run for each model. 
    control_path == the path to the control run
    pulse_path == the path to the pulse run
    m == the model name
    naming convention for the models is MODELID, but if there are multiple realizations it is MODELID_rxix where x is the realization information. eg: CANESM5_r1p1'''
    ds_control = xr.open_mfdataset(control_path, use_cftime=True) #open the control run
    ds_pulse = xr.open_mfdataset(pulse_path, use_cftime=True) #open the pulse run
    lat_corners = cfxr.bounds_to_vertices(ds_control.isel(time = 0)['lat_bnds'], "bnds", order=None) #find the lat corners/bounds
    lon_corners = cfxr.bounds_to_vertices(ds_control.isel(time = 0)['lon_bnds'], "bnds", order=None) #find the lon corners/bounds
    ds_control = ds_control.assign(lon_b=lon_corners, lat_b=lat_corners)

    lat_corners = cfxr.bounds_to_vertices(ds_pulse.isel(time = 0)['lat_bnds'], "bnds", order=None) #find the lat corners/bounds
    lon_corners = cfxr.bounds_to_vertices(ds_pulse.isel(time = 0)['lon_bnds'], "bnds", order=None) #find the lon corners/bounds
    ds_pulse = ds_pulse.assign(lon_b=lon_corners, lat_b=lat_corners)

    if ds_control.attrs['parent_source_id'] != ds_pulse.attrs['parent_source_id']: #check that we are bringing in a control and pulse run from the same model id
        print('WARNING: Control and Pulse runs are not from the same parent source!')
    
    #fix the time for two of the models
    if m == 'NORESM2':
        ds_pulse['time'] = ds_pulse['time']+timedelta(365*1) # NORESM2 pulse time is off by a year as compared to what the documentation says it should be
    if m =='CANESM5_r1p2' or m == 'CANESM5_r2p2' or m == 'CANESM5_r3p2':
        ds_control['time'] = ds_control['time']-timedelta(365*300) # NORESM2 control time is off by a 300 years as compared to what the documentation says it should be
    #select only the times that match up with the pulse
    ds_control = ds_control.sel(time = slice(ds_control['time'].min(), ds_pulse['time'].max()))

    return(ds_control, ds_pulse)


def _regrid_cont_pulse(ds_control, ds_pulse, ds_out):
    '''Regrid the control run and pulse run to our chosen output lat/lon size.
    ds_control == the control run
    ds_pulse == the pulse run
    ds_out == the size of the grid you want, must have lat, lon, lat_b, lon_b
    often this is performed right after _import_combine_pulse_control'''
    
    regridder = xe.Regridder(ds_control, ds_out, "conservative")
    attrs = ds_control.attrs
    ds_control = regridder(ds_control) 
    ds_control.attrs = attrs
    
    regridder = xe.Regridder(ds_pulse, ds_out, "conservative")
    attrs = ds_pulse.attrs
    ds_pulse = regridder(ds_pulse) 
    ds_pulse.attrs = attrs
    
    return(ds_control, ds_pulse)


def _calc_greens(ds_control, ds_pulse, variable, m, pulse_type, climatology, internal_variability_test, pulse_size = 100):
    '''Calculate the Green's Function. 
    ds_control == the control run
    ds_pulse == the pulse run
    variable == the variable on which you want to calculate the Green's Function (eg: tas)
    m == the model name. naming convention for the models is MODELID, but if there are multiple realizations it is MODELID_rxix where x is the realization information. eg: CANESM5_r1p1
    pulse_type == cdr or pulse (which type of pulse is it from the CDRMIP simulations)
    climatology == True or False. If it is true the Greens Function will actually be climatology based, this is best used for evaluating the role of internal variability, but is not necessary for the main Green's Function. This is autoset to False.
    internal_variability_test == a way to test the role of internal variability by time shifting the control run for 5 year intervals from 0-100. This is not needed for the main Green's Function and is auto-set to False
    pulse_size == the size of the GtC pulse. In our case, it was 100. '''
    
    if climatology == False and internal_variability_test == False:
        print('normal run')
        G = (ds_pulse[variable] - ds_control[variable])/(pulse_size) #the key thing: subtract the control from the pulse run and divide by size of the pulse
        times = G.time.get_index('time')
        weights = times.shift(-1, 'MS') - times.shift(1, 'MS') #temporal weights as we are going to take the annual mean rather than working by month
        weights = xr.DataArray(weights, [('time', G['time'].values)]).astype('float')
        G =  (G * weights).groupby('time.year').sum('time')/weights.groupby('time.year').sum('time')
        #select ten years in for two of the models-- these models have 10 years of data before the pulse occurs
        if pulse_type == 'pulse': 
            ten_years_in = 10 #in years
            if m == 'ACCESS': 
                G = G.isel(year = slice(ten_years_in,len(G.year)))
            if m == 'UKESM1_r1':
                G = G.isel(year = slice(ten_years_in,len(G.year)))
        elif pulse_type == 'cdr':
            ten_years_in = 10 #in years
            if m == 'ACCESS':
                G = G.isel(year = slice(ten_years_in,len(G.year)))
        G.attrs = ds_pulse.attrs
        return(G)
    
    elif climatology == True:
        print('climatology run')
        G = (ds_pulse[variable].groupby("time.month") - ds_control[variable].groupby('time.month').mean('time'))/(pulse_size) #the key thing: subtract the control from the pulse run and divide by size of the pulse
        times = G.time.get_index('time') 
        weights = times.shift(-1, 'MS') - times.shift(1, 'MS') #temporal weights as we are going to take the annual mean rather than working by month
        weights = xr.DataArray(weights, [('time', G['time'].values)]).astype('float')
        G =  (G * weights).groupby('time.year').sum('time')/weights.groupby('time.year').sum('time')
        #select ten years in for two of the models-- these models have 10 years of data before the pulse occurs
        if pulse_type == 'pulse':
            ten_years_in = 10 #in years
            if m == 'ACCESS':
                G = G.isel(year = slice(ten_years_in,len(G.year)))
            if m == 'UKESM1_r1':
                G = G.isel(year = slice(ten_years_in,len(G.year)))
        elif pulse_type == 'cdr':
            ten_years_in = 10 #in years
            if m == 'ACCESS':
                G = G.isel(year = slice(ten_years_in,len(G.year)))
        G.attrs = ds_pulse.attrs
        return(G)
    
    elif internal_variability_test == True:
        print('internal variability run')
        G = {}
        for n in np.arange(0,100)[::5]: 
            G[n] = (ds_pulse[variable] - ds_control[variable].shift(time = -n))/(pulse_size) #the key thing: subtract the control from the pulse run and divide by size of the pulse 
            #shift the start time of the control run by every five years from 0-100, so we can quantify the role of internal variability (assuming that every 5 years is approximately a time scale by which variability would occur in the system, essentially a fake way of having more pulse to control runs)
            times = G[n].time.get_index('time') #temporal weights as we are going to take the annual mean rather than working by month
            weights = times.shift(-1, 'MS') - times.shift(1, 'MS')
            weights = xr.DataArray(weights, [('time', G[n]['time'].values)]).astype('float')
            G[n] =  (G[n] * weights).groupby('time.year').sum('time')/weights.groupby('time.year').sum('time')
            #select ten years in for two of the models-- these models have 10 years of data before the pulse occurs
            if pulse_type == 'pulse':
                ten_years_in = 10 #in years
                if m == 'ACCESS':
                    G[n] = G[n].isel(year = slice(ten_years_in,len(G[n].year)))
                if m == 'UKESM1_r1':
                    G[n] = G[n].isel(year = slice(ten_years_in,len(G[n].year)))
            elif pulse_type == 'cdr':
                ten_years_in = 10 #in years
                if m == 'ACCESS':
                    G[n] = G[n].isel(year = slice(ten_years_in,len(G[n].year)))
            G[n].attrs = ds_pulse.attrs
        G = xr.concat([G[m] for m in G.keys()], pd.Index([m for m in G.keys()], name='pulse_year'))
        return(G)    

    

#full function
def import_regrid_calc(control_path, pulse_path, ds_out, variable, m, pulse_type, pulse_size = 100, regrid = True, climatology = False, internal_variability_test = False):
    '''Imports the control run and pulse run for a CMIP6 model run, combines them on the date the pulse starts
    Regrids it to the chosen grid size
    Calculates the Green's Function
    control_path == the path to the control run
    pulse_path == the path to the pulse run
    ds_out == the size of the grid you want, must have lat, lon, lat_b, lon_b
    variable == the variable on which you want to calculate the Green's Function (eg: tas)
    m == the model name. naming convention for the models is MODELID, but if there are multiple realizations it is MODELID_rxix where x is the realization information. eg: CANESM5_r1p1
    pulse_type == cdr or pulse (which type of pulse is it from the CDRMIP simulations)
    climatology == True or False (auto to False). If it is true the Greens Function will actually be climatology based, this is best used for evaluating the role of internal variability, but is not necessary for the main Green's Function. This is autoset to False.
    internal_variability_test == True or False (auto to False). A way to test the role of internal variability by time shifting the control run for 5 year intervals from 0-100. This is not needed for the main Green's Function and is auto-set to False
    pulse_size == the size of the GtC pulse (auto set to 100) 
    regrid == True or False (auto to True). You may be able to skip the regridding of ds_control and ds_pulse, if you're using one model.'''
    
    ds_control, ds_pulse = _import_combine_pulse_control(control_path, pulse_path, m)
    if regrid == True:
        ds_control, ds_pulse = _regrid_cont_pulse(ds_control, ds_pulse, ds_out)
    G = _calc_greens(ds_control, ds_pulse, variable, m, pulse_type, climatology, internal_variability_test, pulse_size)
    return(ds_control, ds_pulse, G)

    
def import_polyfit_G(G_ds_path, G_cdr_ds_path):
    G_ds = xr.open_dataset(G_ds_path)['__xarray_dataarray_variable__']

    G_CDR_ds = xr.open_dataset(G_cdr_ds_path)['__xarray_dataarray_variable__']

    #4th order polyfit
    Gpoly = G_ds.polyfit('year', 4)
    G_ds= xr.polyval(G_ds.year, Gpoly)['polyfit_coefficients']

    Gpoly_cdr = G_CDR_ds.polyfit('year', 4)
    G_CDR_ds= xr.polyval(G_CDR_ds.year, Gpoly_cdr)['polyfit_coefficients']

    G_ds = xr.concat([G_ds, -G_CDR_ds], pd.Index(['pulse','cdr'], name = 'pulse_type'))


    G_ds.name = 'G[tas]'
    G_ds = G_ds.rename({'year':'s'})
    return(G_ds)

#### single regridder ####
def _regrid_ds(ds_in, ds_out):
    regridder = xe.Regridder(ds_in, ds_out,  'conservative', ignore_degenerate = True)
    ds_new = regridder(ds_in) 
    ds_new.attrs = ds_in.attrs
    return(ds_new)


### define our output grid size

ds_out = xr.Dataset(
    {
        "lat": (["lat"], np.arange(-89.5, 90.5, 1.0)),
        "lon": (["lon"], np.arange(0, 360, 1)),
        "lat_b": (["lat_b"], np.arange(-90.,91.,1.0)),
        "lon_b":(["lon_b"], np.arange(.5, 361.5, 1.0))
    }
)


#### function to find area of a grid cell from lat/lon ####
def find_area(ds, R = 6378.1):
    """ ds is the dataset, i is the number of longitudes to assess, j is the number of latitudes, and R is the radius of the earth in km. 
    Must have the ds['lat'] in descending order (90...-90)
    Returns Area of Grid cell in km"""
    circumference = (2*np.pi)*R
    deg_to_m = (circumference/360) 
    dy = (ds['lat_b'].roll({'lat_b':-1}, roll_coords = False) - ds['lat_b'])[:-1]*deg_to_m

    dx1 = (ds['lon_b'].roll({'lon_b':-1}, roll_coords = False) - 
           ds['lon_b'])[:-1]*deg_to_m*np.cos(np.deg2rad(ds['lat_b']))
    
    dx2 = (ds['lon_b'].roll({'lon_b':-1}, roll_coords = False) - 
           ds['lon_b'])[:-1]*deg_to_m*np.cos(np.deg2rad(ds['lat_b'].roll({'lat_b':-1}, roll_coords = False)[:-1]))
    
    A = .5*(dx1+dx2)*dy
    
    #### assign new lat and lon coords based on the center of the grid box instead of edges ####
    A = A.assign_coords(lon_b = ds.lon.values,
                    lat_b = ds.lat.values)
    A = A.rename({'lon_b':'lon','lat_b':'lat'})

    A = A.transpose()
    
    return(A)

A = find_area(ds_out)

def diff_lists(list1, list2):
    return list(set(list1).symmetric_difference(set(list2)))  # or return list(set(list1) ^ set(list2))


########################## model weights ###########################

## We weight by model, because we don't want bias within one model to make it so that our Green's Function over-weights one model that has multiple realizations ##

#define our weights for convolution
model_weights = {'UKESM1_r1': 0.25, 'UKESM1_r2': 0.25, 'UKESM1_r3': 0.25, 'UKESM1_r4': 0.25, 'NORESM2': 1, 'GFDL': 1,
       'MIROC': 1, 'ACCESS': 1,  'CANESM5_r2p2':1/3, 'CANESM5_r1p2':1/3, 'CANESM5_r3p2':1/3}
model_weights = xr.DataArray(
    data=list(model_weights.values()),
    dims=["model"],
    coords=dict(
        model=(["model"], list(model_weights.keys()))
    ),
    attrs=dict(
        description="weights for models"
    ),
)


#define our weights 1pct models
onepct_model_weights = {'UKESM1_r1': 0.25, 'UKESM1_r2': 0.25, 'UKESM1_r3': 0.25, 'UKESM1_r4': 0.25, 'NORESM2': 1, 'GFDL': 1,
       'MIROC': 1, 'CANESM5_r3p1':1/6, 'ACCESS':1, 'CANESM5_r2p2':1/6, 'CANESM5_r2p1':1/6,
       'CANESM5_r1p2':1/6, 'CANESM5_r1p1':1/6, 'CANESM5_r3p2':1/6}
onepct_model_weights = xr.DataArray(
    data=list(onepct_model_weights.values()),
    dims=["model"],
    coords=dict(
        model=(["model"], list(onepct_model_weights.keys()))
    ),
    attrs=dict(
        description="weights for 1pct models"
    ),
)
#define our weights for G
G_model_weights = {'UKESM1_r1': 1, 'NORESM2': 1, 'GFDL': 1,
       'MIROC': 1, 'ACCESS': 1,  'CANESM5_r1p2':1/3, 'CANESM5_r2p2':1/3, 'CANESM5_r3p2':1/3}
G_model_weights = xr.DataArray(
    data=list(G_model_weights.values()),
    dims=["model"],
    coords=dict(
        model=(["model"], list(G_model_weights.keys()))
    ),
    attrs=dict(
        description="weights for Green's function"
    ),
)

#define our weights for the pictrl
pictrl_model_weights = {'UKESM1_r1': 1, 'NORESM2': 1, 'GFDL': 1,
       'MIROC': 1, 'ACCESS': 1,  'CANESM5_r1p1':1/2, 'CANESM5_r1p2':1/2}
pictrl_model_weights = xr.DataArray(
    data=list(pictrl_model_weights.values()),
    dims=["model"],
    coords=dict(
        model=(["model"], list(pictrl_model_weights.keys()))
    ),
    attrs=dict(
        description="weights for models"
    ),
)


######################## dataset dictionaries ###########################


model_run_pulse_dict = {'UKESM1_r1':'UKESM1-0-LL_esm-pi-CO2pulse_r1i1p1f2*', 
                        'MIROC':'MIROC-ES2L_esm-pi-CO2pulse_r1i1p1f2*', 
                        'NORESM2':'NorESM2-LM_esm-pi-CO2pulse_r1i1p1f1*', 
                  'ACCESS':'ACCESS-ESM1-5_esm-pi-CO2pulse_r1i1p1f1*',  
                        'GFDL': 'GFDL-ESM4_esm-pi-CO2pulse_r1i1p1f1**',
                       'CANESM5_r1p2':'CanESM5_esm-pi-CO2pulse_r1i1p2f1*',
                       'CANESM5_r2p2':'CanESM5_esm-pi-CO2pulse_r2i1p2f1*',
                       'CANESM5_r3p2':'CanESM5_esm-pi-CO2pulse_r3i1p2f1*'}

model_run_cdr_pulse_dict = {'UKESM1_r1':'UKESM1-0-LL_esm-pi-cdr-pulse_r1i1p1f2*', 
                        'MIROC':'MIROC-ES2L_esm-pi-cdr-pulse_r1i1p1f2*', 
                        'NORESM2':'NorESM2-LM_esm-pi-cdr-pulse_r1i1p1f1*', 
                  'ACCESS':'ACCESS-ESM1-5_esm-pi-cdr-pulse_r1i1p1f1*',  
                        'GFDL': 'GFDL-ESM4_esm-pi-cdr-pulse_r1i1p1f1**',
                       'CANESM5_r1p2':'CanESM5_esm-pi-cdr-pulse_r1i1p2f1*',
                       'CANESM5_r2p2':'CanESM5_esm-pi-cdr-pulse_r2i1p2f1*',
                       'CANESM5_r3p2':'CanESM5_esm-pi-cdr-pulse_r3i1p2f1*'}


model_run_esm_picontrol_dict = {'UKESM1_r1':'UKESM1-0-LL_esm-piControl_r1i1p1f2*', 
                          'MIROC':'MIROC-ES2L_esm-piControl_r1i1p1f2*', 
                          'NORESM2':'NorESM2-LM_esm-piControl_r1i1p1f1*', 
                  'ACCESS':'ACCESS-ESM1-5_esm-piControl_r1i1p1f1*', 
                          'GFDL': 'GFDL-ESM4_esm-piControl_r1i1p1f1**',
                         'CANESM5_r1p2':'CanESM5_esm-piControl_r1i1p2f1*',
                          'CANESM5_r1p1':'CanESM5_esm-piControl_r1i1p1f1*',
                         } ## for use with pulse run

model_run_picontrol_dict = {'UKESM1_r1':'UKESM1-0-LL_piControl_r1i1p1f2*', 
                          'MIROC':'MIROC-ES2L_piControl_r1i1p1f2*', 
                          'NORESM2':'NorESM2-LM_piControl_r1i1p1f1*', 
                  'ACCESS':'ACCESS-ESM1-5_piControl_r1i1p1f1*', 
                          'GFDL': 'GFDL-ESM4_piControl_r1i1p1f1**',
                         'CANESM5_r1p2':'CanESM5_piControl_r1i1p2f1*',
                          'CANESM5_r1p1':'CanESM5_piControl_r1i1p1f1*',
                         } ## for use with 1pct run


model_run_1pct_dict = {'UKESM1_r1':'UKESM1-0-LL_1pctCO2_r1i1p1f2*',
                       'UKESM1_r2':'UKESM1-0-LL_1pctCO2_r2i1p1f2*',
                       'UKESM1_r3':'UKESM1-0-LL_1pctCO2_r3i1p1f2*',
                       'UKESM1_r4':'UKESM1-0-LL_1pctCO2_r4i1p1f2*',
                        'MIROC':'MIROC-ES2L_1pctCO2_r1i1p1f2*', 
                        'NORESM2':'NorESM2-LM_1pctCO2_r1i1p1f1*', 
                  'ACCESS':'ACCESS-ESM1-5_1pctCO2_r1i1p1f1*',  
                        'GFDL': 'GFDL-ESM4_1pctCO2_r1i1p1f1**',
                      'CANESM5_r1p2':'CanESM5_1pctCO2_r1i1p2f1*',
                      'CANESM5_r2p2':'CanESM5_1pctCO2_r2i1p2f1*',
                      'CANESM5_r3p2':'CanESM5_1pctCO2_r3i1p2f1*',
                      'CANESM5_r1p1':'CanESM5_1pctCO2_r1i1p1f1*',
                      'CANESM5_r2p1':'CanESM5_1pctCO2_r2i1p1f1*',
                      'CANESM5_r3p1':'CanESM5_1pctCO2_r3i1p1f1*'}

model_run_hist_dict = {'CANESM5_r1p1':'CanESM5-1_historical_r1i1p1f1*'}
model_run_control_dict = {'CANESM5_r1p1':'CanESM5-1_piControl_r1i1p1f1*'}
model_run_ssp245_dict = {'CANESM5_r1p1':'CanESM5-1_ssp245_r1i1p1f1*'}
model_run_ssp245_GHG_dict = {'CANESM5_r1p1':'CanESM5_ssp245-GHG_r1i1p1f1*'}
model_run_ssp245_nat_dict = {'CANESM5_r1p1':'CanESM5_ssp245-nat_r1i1p1f1*'}


################## colors ######################

type_color = {'model_1pct': 'darkcyan',
              'model_1000gtc': 'darkcyan',
             'emulator_1pct': 'maroon',
              'emulator_1000gtc':'maroon'}

model_color = {'UKESM1_r1':'darkgreen', 'UKESM1_r2':'mediumaquamarine', 'UKESM1_r3':'seagreen', 'UKESM1_r4':'lightgreen', 'NORESM2':'blue', 'GFDL':'red', 'MIROC':'purple', 'ACCESS':'pink', 'CANESM5_r1p2':'orange', 'CANESM5_r2p2':'sienna', 'CANESM5_r3p2':'goldenrod', 'CANESM5_r1p1':'sienna','mean':'black'}


proper_names = {'UKESM1_r1':'UKESM1', 'MIROC':'MIROC', 'NORESM2':'NORESM2', 'ACCESS':'ACCESS', 'GFDL':'GFDL', 'CANESM5_r1p2':'CANESM5'}