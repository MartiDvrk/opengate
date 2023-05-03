#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate
from scipy.spatial.transform import Rotation
import os
import itk

paths = gate.get_default_test_paths(__file__, "gate_test044_pbs")
output_path = paths.output / "output_test044_sigma_gauss"

# create the simulation
sim = gate.Simulation()

# main options
ui = sim.user_info
ui.g4_verbose = False
ui.g4_verbose_level = 1
ui.visu = False
ui.random_seed = 123654789
ui.random_engine = "MersenneTwister"

# units
km = gate.g4_units("km")
cm = gate.g4_units("cm")
mm = gate.g4_units("mm")
um = gate.g4_units("um")
MeV = gate.g4_units("MeV")
Bq = gate.g4_units("Bq")
nm = gate.g4_units("nm")
deg = gate.g4_units("deg")
mrad = gate.g4_units("mrad")

# add a material database
sim.add_material_database(paths.gate_data / "HFMaterials2014.db")

# RBE lookup table
lookup_tab = "/home/fava/opengate/opengate/data/NIRS_MKM_reduced_data.txt"

#  change world size
world = sim.world
world.size = [600 * cm, 500 * cm, 500 * cm]

# phantoms
m = Rotation.identity().as_matrix()

phantom = sim.add_volume("Box", "phantom_a_1")
phantom.size = [100 * mm, 100 * mm, 300 * mm]
phantom.translation = [0 * mm, 0 * mm, 150 * mm]
phantom.rotation = m
phantom.material = "G4_WATER"
phantom.color = [1, 0, 1, 1]

# default source for tests (from test42)
source = sim.add_source("PencilBeamSource", "mysource1")
source.energy.type = "gauss"
source.energy.mono = 1440 * MeV
source.particle = "ion 6 12"
source.position.type = "disc"
# rotate the disc, equiv to : rot1 0 1 0 and rot2 0 0 1
source.direction.type = "momentum"
source.n = 20000
# source.position.sigma_x = 8 * mm
# source.position.sigma_y = 8 * mm
source.direction.partPhSp_x = [
    2.3335754 * mm,
    2.3335754 * mrad,
    0.00078728 * mm * mrad,
    0,
]
source.direction.partPhSp_y = [
    1.96433431 * mm,
    0.00079118 * mrad,
    0.00249161 * mm * mrad,
    0,
]

# add dose actor
dose = sim.add_actor("RBEActor", "doseInYZ_1")
filename = "phantom_a_1.mhd"
dose.lookup_table_path = lookup_tab
dose.output = output_path / filename
dose.mother = "phantom_a_1"
dose.size = [100, 100, 1500]
dose.spacing = [1, 1, 0.2]
dose.dose_average = True
dose.hit_type = "random"
dose.model = "mkm"


# add stat actor
s = sim.add_actor("SimulationStatisticsActor", "Stats")
s.track_types_flag = True

# physics
p = sim.get_physics_user_info()
p.physics_list_name = "FTFP_INCLXX_EMZ"
sim.set_cut("world", "all", 1000 * km)


# create output dir, if it doesn't exist
if not os.path.isdir(output_path):
    os.mkdir(output_path)

# start simulation
output = sim.start()

# print results at the end
stat = output.get_actor("Stats")
print(stat)
