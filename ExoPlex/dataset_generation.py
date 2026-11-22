import os
import sys
# import ExoPlex as exo
from Examples.generation_conf import Pressure_range_mantle_LM, Pressure_range_mantle_UM, Temperature_range_mantle_UM, Temperature_range_mantle_LM, resolution_LM, \
resolution_UM, Mantle_potential_temp, water_potential_temp, num_mantle_layers, num_core_layers, number_h2o_layers, verbose, use_grids, SOLUTIONS_DIR
from ExoPlex import make_grids, functions, run_perplex
import multiprocessing as mp
import numpy as np
from scipy.stats import qmc
import pandas as pd
import h5py

_CORE_GRID = None
_WATER_GRID = None
_WATER_PHASES = None
CaMg = None
AlMg = None

def initial_worker():
    global _CORE_GRID, _WATER_GRID, _WATER_PHASES, CaMg, AlMg
    _CORE_GRID = make_grids.make_core_grid()
    _WATER_GRID, _WATER_PHASES = make_grids.make_water_grid()
    CaMg = 0.07
    AlMg = 0.09


def run_one_planet(task):
    sample_id, mass, femg, simg, feo, wmf = task

    comp_params = {
        'wt_frac_water': wmf,
        'wt_frac_FeO_wanted': feo,
        'SiMg': simg,
        'FeMg': femg,
        'CaMg': CaMg,
        'AlMg': AlMg,
        'wt_frac_Si_core': 0.,
        'wt_frac_O_core': 0.,
        'wt_frac_S_core': 0.0695,
        'combine_phases': True,
        'use_grids': use_grids,
        'conserve_oxy': False
        }

    structure_params = {
        'Pressure_range_mantle_UM': Pressure_range_mantle_UM,
        'Temperature_range_mantle_UM': Temperature_range_mantle_UM,
        'resolution_UM': resolution_UM,
        'Pressure_range_mantle_LM': Pressure_range_mantle_LM,
        'Temperature_range_mantle_LM': Temperature_range_mantle_LM,
        'resolution_LM': resolution_LM,
        'Mantle_potential_temp': Mantle_potential_temp,
        'water_potential_temp': water_potential_temp,

    }

    layers = [num_mantle_layers,num_core_layers,number_h2o_layers]

    try:
        core_wt_per, mantle_wt_per, _, core_mass_frac = functions.get_percents(comp_params, False)

        if core_mass_frac < 0.005:
            return None

        if use_grids == True:
            filename = functions.find_filename(comp_params, verbose)
        else:
            filename=''

        mantle_filename = run_perplex.run_perplex(*[mantle_wt_per,comp_params,structure_params,filename,verbose, True])
        grids_UM, names_UM = make_grids.make_mantle_grid(mantle_filename, mantle_wt_per, True, use_grids)
        names_UM.append('Fe')

        mantle_filename = run_perplex.run_perplex(*[mantle_wt_per,comp_params,structure_params,filename,verbose,False])
        grids_LM, names_LM = make_grids.make_mantle_grid(mantle_filename, mantle_wt_per, False, use_grids)
        names_LM.append('Fe')

        grids = {
            'UM': grids_UM,
            'LM': grids_LM,
            'Core': _CORE_GRID,
            'Water': _WATER_GRID
        }

        Planet = functions.find_Planet_mass(mass, core_mass_frac, structure_params, comp_params, grids, core_wt_per, layers, verbose)
        Planet['phase_names_low'] = names_UM
        Planet['phase_names_high'] = names_LM
        Planet['water_phases'] = _WATER_PHASES

        Planet['phases'],Planet['phase_names'] = functions.get_phases(Planet, grids, layers, comp_params['combine_phases'])
        Planet['planet_mass'] = Planet['mass'][-1]
        Planet['planet_radius'] = Planet['radius'][-1]
        Planet['planet_density'] = Planet['mass'][-1] / (4 / 3 * np.pi * pow(Planet['radius'][-1], 3)) / 1000

        Planet['CMF'] = Planet['mass'][num_core_layers - 1] / Planet['mass'][-1]
        Planet['MMF'] = (Planet['mass'][num_core_layers + num_mantle_layers - 1] - Planet['mass'][num_core_layers - 1]) / Planet['mass'][-1]
        Planet['WMF'] = 1 - Planet['CMF'] - Planet['MMF']

        Planet['CRF'] = Planet['radius'][num_core_layers - 1] / Planet['radius'][-1]
        Planet['MRF'] = (Planet['radius'][num_core_layers + num_mantle_layers - 1] - Planet['radius'][num_core_layers - 1]) / Planet['radius'][-1]
        Planet['WRF'] = 1 - Planet['CRF'] - Planet['MRF']

        Planet['PCMB'] = Planet['pressure'][num_core_layers] / 1e4
        Planet['TCMB'] = Planet['temperature'][num_core_layers]

        Planet['sample_id'] = int(sample_id)

        #check to see if solution works (P goes up with depth etc.)
        functions.check(Planet)
        return Planet

    except (AssertionError, ValueError, RuntimeError, SystemExit):
        return None


def run_shared(tasks, n_workers=4):
    ctx = mp.get_context('spawn')
    results = []

    with ctx.Pool(processes=n_workers, initializer=initial_worker) as pool:
        for res in pool.imap(run_one_planet, tasks, chunksize=10):
            if res is not None:
                results.append(res)

    return results


sampler = qmc.Sobol(d=5, scramble=True, seed=42)
u = sampler.random_base2(m=8)
l_bounds = [0.1, 0.0, 0.1, 0.00, 1e-4]  # M, FeMg, SiMg, FeO, WMF
u_bounds = [10.0, 6.0, 2.0, 0.20, 0.10]

samples = qmc.scale(u, l_bounds, u_bounds)
start_id = 0
end_id = len(samples)

t = [
    (idx, *samples[idx]) for idx in range(start_id, end_id)
]

# planets = run_shared(t, n_workers=8)
initial_worker()
p = run_one_planet(t[0])


sample_id, mass, femg, simg, feo, wmf = t[0]

comp_params = {
    'wt_frac_water': wmf,
    'wt_frac_FeO_wanted': feo,
    'SiMg': simg,
    'FeMg': femg,
    'CaMg': CaMg,
    'AlMg': AlMg,
    'wt_frac_Si_core': 0.,
    'wt_frac_O_core': 0.,
    'wt_frac_S_core': 0.0695,
    'combine_phases': True,
    'use_grids': use_grids,
    'conserve_oxy': False
    }

structure_params = {
    'Pressure_range_mantle_UM': Pressure_range_mantle_UM,
    'Temperature_range_mantle_UM': Temperature_range_mantle_UM,
    'resolution_UM': resolution_UM,
    'Pressure_range_mantle_LM': Pressure_range_mantle_LM,
    'Temperature_range_mantle_LM': Temperature_range_mantle_LM,
    'resolution_LM': resolution_LM,
    'Mantle_potential_temp': Mantle_potential_temp,
    'water_potential_temp': water_potential_temp,

}

layers = [num_mantle_layers,num_core_layers,number_h2o_layers]

core_wt_per, mantle_wt_per, _, core_mass_frac = functions.get_percents(comp_params, False)

if core_mass_frac < 0.005:
    print('None')

if use_grids == True:
    filename = functions.find_filename(comp_params, verbose)
else:
    filename=''

mantle_filename = run_perplex.run_perplex(*[mantle_wt_per,comp_params,structure_params,filename,verbose, True])
grids_UM, names_UM = make_grids.make_mantle_grid(mantle_filename, mantle_wt_per, True, use_grids)
names_UM.append('Fe')

mantle_filename = run_perplex.run_perplex(*[mantle_wt_per,comp_params,structure_params,filename,verbose,False])
grids_LM, names_LM = make_grids.make_mantle_grid(mantle_filename, mantle_wt_per, False, use_grids)
names_LM.append('Fe')

grids = {
    'UM': grids_UM,
    'LM': grids_LM,
    'Core': _CORE_GRID,
    'Water': _WATER_GRID
}

Planet = functions.find_Planet_mass(mass, core_mass_frac, structure_params, comp_params, grids, core_wt_per, layers, verbose)
Planet['phase_names_low'] = names_UM
Planet['phase_names_high'] = names_LM
Planet['water_phases'] = _WATER_PHASES

Planet['phases'],Planet['phase_names'] = functions.get_phases(Planet, grids, layers, comp_params['combine_phases'])
Planet['planet_mass'] = Planet['mass'][-1]
Planet['planet_radius'] = Planet['radius'][-1]
Planet['planet_density'] = Planet['mass'][-1] / (4 / 3 * np.pi * pow(Planet['radius'][-1], 3)) / 1000

Planet['CMF'] = Planet['mass'][num_core_layers - 1] / Planet['mass'][-1]
Planet['MMF'] = (Planet['mass'][num_core_layers + num_mantle_layers - 1] - Planet['mass'][num_core_layers - 1]) / Planet['mass'][-1]
Planet['WMF'] = 1 - Planet['CMF'] - Planet['MMF']

Planet['CRF'] = Planet['radius'][num_core_layers - 1] / Planet['radius'][-1]
Planet['MRF'] = (Planet['radius'][num_core_layers + num_mantle_layers - 1] - Planet['radius'][num_core_layers - 1]) / Planet['radius'][-1]
Planet['WRF'] = 1 - Planet['CRF'] - Planet['MRF']

Planet['PCMB'] = Planet['pressure'][num_core_layers] / 1e4
Planet['TCMB'] = Planet['temperature'][num_core_layers]

Planet['sample_id'] = int(sample_id)

#check to see if solution works (P goes up with depth etc.)
functions.check(Planet)









