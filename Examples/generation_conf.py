import os

verbose = False

#create filename to store values
Output_filename = 'Filename'

#Next user must input the ratios by mole (Earth is Ca/Mg = .07, Si.Mg = 0.90, Al/Mg = 0.09, Fe/Mg = 0.9)
CaMg = 0.07
AlMg = 0.09


#Don't forget that if you have water you need to add water layers
number_h2o_layers = 300

#The potential Temperature of Water, if present
water_potential_temp = 300.

#What fraction of the mantle would you like to be made of FeO? This Fe will be pulled from the core.
conserve_oxy = False

#Now we can mix various elements into the core or mantle
wt_frac_Si_core = 0. #by mass <1, note if you conserve oxygen this is calculated for you
wt_frac_O_core = 0. #by mass
wt_frac_S_core = 0.0695 #by mass

#What potential temperature (in K) do you want to start your mantle adiabat?
Mantle_potential_temp = 1600.

#Input the resolution of your upper mantle and lower mantle composition, density grids
#These are input as number of T, P points. 50 50 = 2500 grid points, which takes about
#5 minutes to calculate. Lower mantle resolution does not need to be higher since it's
#mostly ppv.
resolution_UM = '25 75'
resolution_LM = '75 75'

#lastly we need to decide how many layers to put in the planet. This is the resolution of
#the mass-radius sampling.
num_mantle_layers = 300
num_core_layers = 300

comp_keys = ['wt_frac_water','FeMg','SiMg','CaMg','AlMg','wt_frac_FeO_wanted','wt_frac_Si_core',
                          'wt_frac_O_core','wt_frac_S_core', 'combine_phases','use_grids','conserve_oxy']
struct_keys = ['Pressure_range_mantle_UM','Temperature_range_mantle_UM','resolution_UM',
                         'Pressure_range_mantle_LM', 'Temperature_range_mantle_LM', 'resolution_LM',
                         'Mantle_potential_temp','water_potential_temp']

Pressure_range_mantle_UM = '1 1400000'
Temperature_range_mantle_UM = '1600 3500'

Pressure_range_mantle_LM = '1250000 40000000'
Temperature_range_mantle_LM = '1700 7000'

combine_phases = True #note this is automatically done when use_grids = False and cannot be changed
use_grids = True

_EXOPLEX_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLUTIONS_DIR = os.path.join(_EXOPLEX_ROOT, 'Solutions_Small')
BURNMAN_DIR = os.path.join(_EXOPLEX_ROOT, 'ExoPlex', 'burnman', 'data', 'input_masses')