import opengate as gate
import itk
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

## ------ INITIALIZE SIMULATION ENVIRONMENT ---------- ##
paths = gate.get_default_test_paths(__file__, "gate_test044_pbs")
output_path = paths.output / "output_test051_rtp"
ref_path = paths.output_ref / "test051_ref"


# create the simulation
sim = gate.Simulation()

# main options
ui = sim.user_info
ui.g4_verbose = False
ui.g4_verbose_level = 1
ui.visu = False
ui.random_seed = 12365478910
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
rad = gate.g4_units("rad")

# add a material database
sim.add_material_database(paths.gate_data / "HFMaterials2014.db")

## Beamline model
IR2HBL = gate.BeamlineModel()
IR2HBL.name = None
IR2HBL.radiation_types = "ion 6 12"
# Nozzle entrance to Isocenter distance
IR2HBL.distance_nozzle_iso = 1300.00  # 1648 * mm#1300 * mm
# SMX to Isocenter distance
IR2HBL.distance_stearmag_to_isocenter_x = 6700.00
# SMY to Isocenter distance
IR2HBL.distance_stearmag_to_isocenter_y = 7420.00
# polinomial coefficients
IR2HBL.energy_mean_coeffs = [11.91893485094217, -9.539517997860457]
IR2HBL.energy_spread_coeffs = [0.0004790681841295621, 5.253257865904452]
IR2HBL.sigma_x_coeffs = [2.3335753978880014]
IR2HBL.theta_x_coeffs = [0.0002944903217664001]
IR2HBL.epsilon_x_coeffs = [0.0007872786903040108]
IR2HBL.sigma_y_coeffs = [1.9643343053823967]
IR2HBL.theta_y_coeffs = [0.0007911780133478402]
IR2HBL.epsilon_y_coeffs = [0.0024916149017600447]


#  change world size
world = sim.world
world.size = [600 * cm, 500 * cm, 500 * cm]

# nozzle box
box = sim.add_volume("Box", "box")
box.size = [500 * mm, 500 * mm, 1000 * mm]
box.translation = [1148 * mm, 0.0, 0.0]
box.rotation = Rotation.from_euler("y", -90, degrees=True).as_matrix()
box.material = "Vacuum"
box.color = [0, 0, 1, 1]

# nozzle WET
nozzle = sim.add_volume("Box", "nozzle")
nozzle.mother = box.name
nozzle.size = [500 * mm, 500 * mm, 2 * mm]
nozzle.material = "G4_WATER"

# treatment info
# rt_plan_path = "/home/ideal/0_Data/02_ref_RTPlans/01_ref_Plans_CT_RTpl_RTs_RTd/03_AbsDose/01_IR2HBLc/E120_0MeVn/RP1.2.752.243.1.1.20230202162643145.2610.40668_tagman.dcm"
rt_plan_path = "/home/ideal/0_Data/99_NonRefPlans/Box16_15_C_ISD0_Phys/DCM/RP1.2.752.243.1.1.20190416091012952.3300.34040.dcm"
treatment = gate.radiation_treatment(rt_plan_path)
# structs = treatment.structures
# bemaset = treatment.beamset_info
# doses = treatment.rt_doses
ct_image = treatment.ct_image
mhd_ct = str(ref_path / "absolute_dose_ct.mhd")
ct_image.write_to_file(mhd_ct)

# # container
# phantom = sim.add_volume("Box", "phantom")
# phantom.size = [500 * mm, 500 * mm, 400 * mm]
# phantom.rotation = Rotation.from_euler("y", 90, degrees=True).as_matrix()
# phantom.translation = [-200.0, 0.0, 0]
# phantom.material = "G4_WATER"
# phantom.color = [0, 0, 1, 1]

# patient
patient = sim.add_volume("Image", "patient")
patient.image = mhd_ct
# patient.mother = phantom.name
patient.material = "G4_AIR"  # material used by default
patient.voxel_materials = [
    [-1024, -300, "G4_AIR"],
    [-300, 3000, "G4_WATER"],
]
patient.dump_label_image = ref_path / "test_ct_label.mhd"

# physics
p = sim.get_physics_user_info()
p.physics_list_name = "FTFP_INCLXX_EMZ"  #'QGSP_BIC_HP_EMZ' #"FTFP_INCLXX_EMZ"
sim.set_cut("world", "all", 1000 * km)

# add dose actor
dose = sim.add_actor("DoseActor", "doseInXYZ")
dose.output = output_path / "abs_dose_ct.mhd"
dose.mother = patient.name
dose.size = [200, 200, 200]
dose.spacing = [1 * mm, 1 * mm, 1 * mm]
dose.hit_type = "random"
dose.gray = True


## source
nplan = treatment.beamset_info.mswtot
nSim = 20000  # 328935  # particles to simulate per beam
tps = gate.TreatmentPlanSource("RT_plan", sim)
tps.set_beamline_model(IR2HBL)
tps.set_particles_to_simulate(nSim)
tps.set_spots_from_rtplan(rt_plan_path)
tps.initialize_tpsource()

# add stat actor
s = sim.add_actor("SimulationStatisticsActor", "Stats")
s.track_types_flag = True
# start simulation
output = sim.start()

## -------------END SCANNING------------- ##
# print results at the end
stat = output.get_actor("Stats")
print(stat)

# create output dir, if it doesn't exist
if not os.path.isdir(output_path):
    os.mkdir(output_path)

## ------ TESTS -------##
dose_path = gate.scale_dose(
    str(dose.output).replace(".mhd", "_dose.mhd"),
    nplan / nSim,
    output_path / "threeDdoseWater.mhd",
)

# # ABSOLUTE DOSE

# read output and ref
img_mhd_out = itk.imread(dose_path)

# 1D
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(25, 10))
gate.plot_img_axis(ax, img_mhd_out, "x profile", axis="x")
gate.plot_img_axis(ax, img_mhd_out, "y profile", axis="y")
gate.plot_img_axis(ax, img_mhd_out, "z profile", axis="z")

plt.show()
# fig.savefig(output_path / "dose_profiles_water.png")


# gate.test_ok(ok)
