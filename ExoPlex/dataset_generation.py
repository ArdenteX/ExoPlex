import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
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
from tqdm import tqdm

_CORE_GRID = None
_WATER_GRID = None
_WATER_PHASES = None
CaMg = 0.07
AlMg = 0.09
n_layers = num_mantle_layers + num_core_layers + number_h2o_layers
# chunk_size = 10

observable_keys = ['planet_mass', 'planet_radius', 'FeMg', 'SiMg', 'sample_id']
interiors_keys = ['CMF', 'MMF', 'WMF', 'CRF', 'MRF', 'WRF', 'PCMB', 'TCMB', 'sample_id']
out_keys = [
    'layer_mass', 'layer_radius', 'layer_pressure', 'layer_temperature', 'layer_density', 'layer_gravity', 'beta', 'layer_types', 'sample_id'
]


def get_result(planet, femg, simg):
    # p_mass = np.repeat(planet['mass'][-1], n_layers)
    # p_radius = np.repeat(planet['radius'][-1], n_layers)

    # p_cmf = np.repeat(planet['mass'][num_core_layers - 1] / planet['mass'][-1], n_layers)
    # p_mmf = np.repeat((planet['mass'][num_core_layers + num_mantle_layers - 1] - planet['mass'][num_core_layers - 1]) / \
    #                   planet['mass'][-1], n_layers)
    # p_wmf = 1 - p_cmf - p_mmf

    cmf = planet['mass'][num_core_layers - 1] / planet['mass'][-1]
    mmf = (planet['mass'][num_core_layers + num_mantle_layers - 1] - planet['mass'][num_core_layers - 1]) / \
                      planet['mass'][-1]
    wmf = 1 - cmf - mmf

    # p_crf = np.repeat(planet['radius'][num_core_layers - 1] / planet['radius'][-1], n_layers)
    # p_mrf = np.repeat((planet['radius'][num_core_layers + num_mantle_layers - 1] - planet['radius'][
    #     num_core_layers - 1]) / planet['radius'][-1], n_layers)
    # p_wrf = 1 - p_crf - p_mrf

    crf = planet['radius'][num_core_layers - 1] / planet['radius'][-1]
    mrf = (planet['radius'][num_core_layers + num_mantle_layers - 1] - planet['radius'][
        num_core_layers - 1]) / planet['radius'][-1]
    wrf = 1 - crf - mrf

    # p_pcmb = np.repeat(planet['pressure'][num_core_layers] / 1e4, n_layers)
    # p_tcmb = np.repeat(planet['temperature'][num_core_layers], n_layers)
    p_id = np.repeat(planet['sample_id'], n_layers)

    p_layer_types = np.asarray([0] * num_core_layers + [1] * num_mantle_layers + [2] * number_h2o_layers)

    observables = np.asarray([
        planet['mass'][-1],
        planet['radius'][-1],
        femg,
        simg,
        planet['sample_id']
    ])

    interiors = np.asarray([
        cmf,
        mmf,
        wmf,
        crf,
        mrf,
        wrf,
        planet['pressure'][num_core_layers] / 1e4,
        planet['temperature'][num_core_layers],
        planet['sample_id']
    ])

    layer_params = np.stack([
        planet['mass'],
        planet['radius'],
        planet['pressure'],
        planet['temperature'],
        planet['density'],
        planet['gravity'],
        planet['alpha'] / planet['CP'],
        p_layer_types,
        p_id
    ], axis=1)

    return observables, interiors, layer_params


def initial_worker():
    global _CORE_GRID, _WATER_GRID, _WATER_PHASES, CaMg, AlMg, n_layers
    _CORE_GRID = make_grids.make_core_grid()
    _WATER_GRID, _WATER_PHASES = make_grids.make_water_grid()
    CaMg = 0.07
    AlMg = 0.09
    n_layers = num_mantle_layers + num_core_layers + number_h2o_layers


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
        # Planet['planet_mass'] = Planet['mass'][-1]
        # Planet['planet_radius'] = Planet['radius'][-1]
        # Planet['planet_density'] = Planet['mass'][-1] / (4 / 3 * np.pi * pow(Planet['radius'][-1], 3)) / 1000
        #
        # Planet['CMF'] = Planet['mass'][num_core_layers - 1] / Planet['mass'][-1]
        # Planet['MMF'] = (Planet['mass'][num_core_layers + num_mantle_layers - 1] - Planet['mass'][num_core_layers - 1]) / Planet['mass'][-1]
        # Planet['WMF'] = 1 - Planet['CMF'] - Planet['MMF']
        #
        # Planet['CRF'] = Planet['radius'][num_core_layers - 1] / Planet['radius'][-1]
        # Planet['MRF'] = (Planet['radius'][num_core_layers + num_mantle_layers - 1] - Planet['radius'][num_core_layers - 1]) / Planet['radius'][-1]
        # Planet['WRF'] = 1 - Planet['CRF'] - Planet['MRF']
        #
        # Planet['PCMB'] = Planet['pressure'][num_core_layers] / 1e4
        # Planet['TCMB'] = Planet['temperature'][num_core_layers]

        Planet['sample_id'] = int(sample_id)
        functions.check(Planet) # check to see if solution works (P goes up with depth etc.)
        obs, interiors, layers = get_result(Planet, femg, simg)
        return {
            'obs': obs,
            'interiors': interiors,
            'layers': layers,
        }

    except (AssertionError, ValueError, RuntimeError, SystemExit):
        return None


def run_shared(
        tasks,
        n_workers=4,
        flush_every=10,
        ds_chunksize=1024,
        pool_chunksize=3
):
    ctx = mp.get_context('spawn')

    with h5py.File('D:\\Resource\\HPC\\dataset_test\\results_1_100.h5', 'w') as f:
        obs = f.create_group('observables')
        ins = f.create_group('interiors')
        ls = f.create_group('layers')

        x = obs.create_dataset(
            'X', dtype=np.float32,
            shape=(0, len(observable_keys)),
            maxshape=(None, len(observable_keys)),
            chunks=(ds_chunksize, len(observable_keys)),
            compression=None
        )

        y = ins.create_dataset(
            'Y', dtype=np.float32,
            shape=(0, len(interiors_keys)),
            maxshape=(None, len(interiors_keys)),
            chunks=(ds_chunksize, len(interiors_keys)),
            compression=None
        )

        l = ls.create_dataset(
            'L', dtype=np.float32,
            shape=(0, n_layers, len(out_keys)),
            maxshape=(None, n_layers, len(out_keys)),
            chunks=(1, n_layers, len(out_keys)),
            compression=None
        )

        obs['X'].attrs['columns'] = observable_keys
        ins['Y'].attrs['columns'] = interiors_keys
        ls['L'].attrs['columns'] = out_keys

        buf_x, buf_y, buf_l = [], [], []

        def flush_buffer():
            if not buf_x:
                return

            old_size = x.shape[0]
            new_size = old_size + len(buf_x)
            x.resize(new_size, axis=0)
            y.resize(new_size, axis=0)
            l.resize(new_size, axis=0)

            x[old_size: new_size] = np.stack(buf_x, axis=0)
            y[old_size: new_size] = np.stack(buf_y, axis=0)
            l[old_size: new_size] = np.stack(buf_l)

            f.flush()
            buf_x.clear()
            buf_y.clear()
            buf_l.clear()

        valid_cnt = 0
        fail_cnt = 0
        try:
            with ctx.Pool(processes=n_workers, initializer=initial_worker) as pool:
                pbar = tqdm(
                    pool.imap_unordered(run_one_planet, tasks, chunksize=pool_chunksize),
                    total=len(tasks),
                    desc="Generating planets",
                    unit="planet",
                    mininterval=1.0  # 上 Slurm 非交互跑大任务时可调成 5.0 或 10.0
                )
                for res in pbar:
                    if res is not None:
                        buf_x.append(res['obs'])
                        buf_y.append(res['interiors'])
                        buf_l.append(res['layers'])
                        valid_cnt += 1

                        if len(buf_x) >= flush_every:
                            flush_buffer()
                    else:
                        fail_cnt += 1

                    pbar.set_postfix(valid=valid_cnt, fail=fail_cnt, saved=x.shape[0])
        finally:
            flush_buffer()

    return 'ok'


if __name__ == '__main__':
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

    # planets = run_shared(t[:100], n_workers=8)
    run_shared(t[:100], n_workers=8)









