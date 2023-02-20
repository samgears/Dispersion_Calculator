# import valid libaries
import matplotlib.pyplot as plt
import matplotlib.pyplot as py
import numpy as np
import gdsfactory as gf
import gdsfactory.simulation.gmeep as gm
import gdsfactory.simulation as sim
import gdsfactory.simulation.modes as gm
import meep as mp
import sys
from datetime import datetime
import scipy
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
import math
import os
import csv
import time
from tqdm import tqdm

# Sellmeir fits
def s(S, T):
    return S[0]+S[1]*T+S[2]*T**2+S[3]*T**3+S[4]*T**4

def l(L, T):
    return L[0]+L[1]*T+L[2]*T**2+L[3]*T**3+L[4]*T**4

def index(lam, T, S1, S2, S3, L1, L2, L3):
    return np.sqrt(((s(S1, T)*lam**2)/(lam**2-l(L1, T)**2))+((s(S2, T)*lam**2)/(lam**2-l(L2, T)**2))+((s(S3, T)*lam**2)/(lam**2-l(L3, T)**2))+1)

# Sellemir coefficents
S1_si = [10.4907, -2.08020e-04, 4.21694e-06, -5.82298e-09, 3.44688e-12]
S2_si = [-1346.61, 29.1664, -0.278724, 1.05939e-03, -1.35089e-06]
S3_si = [4.42827e7, -1.76213e6, -7.61575e4, 678.414, 103.243]
L1_si = [0.299713, -1.14234e-5, 1.67134e-7, -2.51049e-10, 2.32484e-14]
L2_si = [-3.51710e3, 42.3892, -0.357957, 1.17504e-3, -1.13212e-6]
L3_si = [1.71400e6, -1.44984e5, -6,90744e3, -39.3699, 23.5770]

# Do the same thing for the Silica from this paper: https://arxiv.org/pdf/0805.0091.pdf
# Uses corning fused silica, should be fine, check with AMF data at 295K
S1_sio2 = [1.10127, -4.94251e-5, 5.27414e-7, -1.59700e-9, 1.75949e-12]
S2_sio2 = [1.78752e-5, 4.76391e-5, -4.49019e-7, 1.44546e-9, -1.57223e-12]
S3_sio2 = [7.93552e-1, -1.27815e-3,  1.84595e-5, -9.20275e-8, 1.48829e-10]
L1_sio2 = [-8.90600e-2, 9.08730e-6, -6.53638e-8, 7.77072e-11, 6.84605E-14]
L2_sio2 = [2.97562e-1, -8.59578e-4, 6.59069e-6, -1.09482e-8, 7.85145e-13]
L3_sio2 = [9.34454, -7.09788e-3, 1.01968e-4, -5.07660e-7, 8.21348E-10]


# function that returns refractive indexes for given wavelength and temperature
def get_si_si02_n(wavelength_lambda,desired_temp=300):

    n_lambda_si = 0
    n_lambda_sio2 = 0

    n_lambda_si = index(wavelength_lambda, desired_temp, S1_si, S2_si, S3_si, L1_si, L2_si, L3_si)
    n_lambda_sio2 = index(wavelength_lambda, desired_temp, S1_sio2, S2_sio2, S3_sio2, L1_sio2, L2_sio2, L3_sio2)
    
    return [n_lambda_si,n_lambda_sio2]

# omega lambda conversions
def lambda_omega_conversion(parameter):
    return 2 * np.pi * 299792458 / parameter

# calculate effective indexes for a given geometry and centre wavelength
def get_n_eff(n_lambda_si,n_lambda_sio2,centre_lambda,mode,side_angles=15,wg_width= 0.51,wg_thickness=0.22,resolution=100,sz=3,sy=3):

    refractive_eff_index = np.zeros(mode)
    ratio_nl_s = np.zeros(mode)
    
    # find the supported modes
    modes = gm.find_modes_waveguide(
        wg_width= wg_width,
        wavelength=centre_lambda,
        ncore=n_lambda_si,
        nclad=n_lambda_sio2,
        wg_thickness=wg_thickness,
        resolution=resolution,
        sz=sz,
        sy=sy,
        nmodes=mode,
        sidewall_angle=side_angles,
        parity=mp.ODD_Y + mp.EVEN_Z # for a TE source
    )
    # sweep over the modes to find the effective areas and effective indexes
    for i in range(mode):
        # import data
        refractive_eff_index[i] = modes[i+1].neff
        E = modes[i+1].E
        H = modes[i+1].H
        # calculate poynting vector
        cross_product = np.cross(E,H)  # Calculate cross product of two matrices E and H
        cross_product_z = cross_product[:,:,:,0]
        cross_product_z = cross_product_z[:,:,0]

        # get the refractive index
        n_raw = modes[i+1].eps
        # transpose mask to match up, annoying but for some reason klayout requires this
        n_raw = n_raw.T
        n_raw_max = np.max(n_raw)
        n_raw_min = np.min(n_raw)
        n_mask_nl = (n_raw - n_raw_min)/(n_raw_max-n_raw_min)
        # calculate nonlinear areas
        sz_nl = np.sum((np.multiply(cross_product_z,n_mask_nl)))
        sz_all = np.sum((cross_product_z))
        ratio_nl_s[i] = (sz_all/sz_nl)*(np.sum(n_mask_nl)/(np.shape(n_mask_nl)[0]*np.shape(n_mask_nl)[1]))*(sz*sy)

    return refractive_eff_index,ratio_nl_s

# does a wavelength sweep to find n and aeff
def get_neff_aeff(mode,side_angles,wg_thickness,wg_width,resolution,sy_base,waveguide_temperature,start_wavelength,end_wavelength,w_k_sweep_steps):
    # mode solved datapoints
    # simulation structure parameters
    u = 1E-6
    # simulation bounding size
    sy = np.round((sy_base + wg_width),2)
    sz = sy

    # convert these values to frequency
    # we are sweeping linear in frequency, nonlinear in wavelength
    start_frequency = lambda_omega_conversion(start_wavelength)
    end_frequency = lambda_omega_conversion(end_wavelength)
    sample_frequencys = np.linspace(start_frequency,end_frequency,w_k_sweep_steps)
    sample_wavelengths = lambda_omega_conversion(sample_frequencys)

    # final effective indecies and effective areas
    sample_eff_indexes = np.zeros([w_k_sweep_steps,mode])
    sample_a_eff = np.zeros([w_k_sweep_steps,mode])

    # sweep over all of the wavelengths
    # find n_eff and a_eff for the number of stated modes
    for index, wavelength in enumerate(sample_wavelengths):
        # convert wavelength to um
        print('Starting Wavelength '+str(index)+'/'+str(w_k_sweep_steps))
        wg_start_time = time.time()
        wavelength = wavelength / u

        # get absolute indecies
        eff_indexes = get_si_si02_n(wavelength,desired_temp=waveguide_temperature)
        # get effective indecies
        n_eff_out,a_eff_out = get_n_eff(eff_indexes[0],eff_indexes[1],wavelength,mode,side_angles=side_angles,wg_width=wg_width,wg_thickness=wg_thickness,resolution=resolution,sz=sz,sy=sy)
        # put into arrays
        sample_eff_indexes[index,:] = n_eff_out
        sample_a_eff[index,:] = a_eff_out
        wg_end_time = time.time()
        elapsed_time = wg_end_time - wg_start_time
        print('Finished Wavelength '+str(index)+'/'+str(w_k_sweep_steps) + ' in ' + str(elapsed_time) + 's')

    return sample_wavelengths,sample_eff_indexes,sample_a_eff

# global script
u = 1E-6

# fixed geometry parameters
mode = 2
side_angles = 2
wg_thickness = 0.22
waveguide_temperature = 295

# simulation parameters
resolution = 100
sy_base = 4

# wavelength w k sweep parameters
start_wavelength = 1.8 * u
end_wavelength = 2.2 * u
w_k_sweep_steps = 3

# sweeping waveguide width, in um
wg_initial_width = 0.7
wg_final_width = 2
wg_width_sweep_steps = 3
# make a dictionary of everything
metadata = {
    'mode':str(mode),
    'side_angles':str(side_angles),
    'wg_thickness':str(wg_thickness),
    'waveguide_temperature':str(waveguide_temperature),
    'resolution':str(resolution),
    'sy_base':str(sy_base),
    'start_wavelength':str(start_wavelength),
    'end_wavelength':str(end_wavelength),
    'w_k_sweep_steps':str(w_k_sweep_steps),
    'wg_initial_width':str(wg_initial_width),
    'wg_final_width':str(wg_final_width),
    'wg_width_sweep_steps':str(wg_width_sweep_steps),
    }

write_to_file = True
if (write_to_file):
    # put this into a huge csv file
    folder_name = 'dispersion_sweeps/single_waveguide/headless_server'+str(mode)+'_resolution_'+str(resolution)+'_sybase'+str(sy_base)+'_wg_initial_width'+str(wg_initial_width)+'_wg_final_width'+str(wg_final_width)+'_w_k_sweep_steps'+str(w_k_sweep_steps)
    os.mkdir(folder_name)

    # put everything in a header csv file
    file_header = folder_name+'/header.csv'
    file = open(file_header, "w")

    # write metadata to header file
    writer = csv.writer(file)
    for key, value in metadata.items():
        writer.writerow([key, value])
    file.close()

# start doing waveguide width sweeps
widths = np.linspace(wg_initial_width,wg_final_width,wg_width_sweep_steps)

for index_width, wg_width in enumerate(widths):
    start_time_width = time.time()
    print('Starting Width '+str(index_width)+'/'+str(wg_width_sweep_steps))
    sample_wavelengths,sample_eff_indexes,sample_a_eff = get_neff_aeff(mode,side_angles,wg_thickness,wg_width,resolution,sy_base,waveguide_temperature,start_wavelength,end_wavelength,w_k_sweep_steps)
    # write to file
    print('Writing to csv...')
    if (write_to_file):
        file_name = folder_name +'/' + str(wg_width) + '.csv'
        with open(file_name, 'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(["Wavelength", "mode" , "n_eff", "a_eff / um^2"])
            for i in range(w_k_sweep_steps):
                for j in range(mode):
                    writer.writerow([sample_wavelengths[i], str(j) , sample_eff_indexes[i,j],sample_a_eff[i,j]])
        print('CSV Written')
    end_time_width = time.time()
    print('Finished Width '+str(index_width)+'/'+str(wg_width_sweep_steps) + ' in ' + str((end_time_width-start_time_width)) + 's')


print('Everything Finished!!')