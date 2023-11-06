import opengate as gate
import itk
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation
from opengate.tests import utility
from opengate.contrib.beamlines.ionbeamline import BeamlineModel
from opengate.contrib.tps.ionbeamtherapy import spots_info_from_txt, TreatmentPlanSource

if __name__ == "__main__":
    paths = utility.get_default_test_paths(__file__, "gate_test044_pbs")

    output_path = paths.output / "output_test059_rtp"
    ref_path = paths.output_ref / "test059_ref"

    # create output dir, if it doesn't exist
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

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
    km = gate.g4_units.km
    cm = gate.g4_units.cm
    mm = gate.g4_units.mm
    gcm3 = gate.g4_units.g_cm3

    # add a material database
    sim.add_material_database(paths.gate_data / "HFMaterials2014.db")

    ## Beamline model
    IR2HBL = BeamlineModel()
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

    # source and beamline info
    spots, ntot, energies, gantry_angle = spots_info_from_txt(
        ref_path / "TreatmentPlan4Gate-F5x5cm_E120MeVn.txt", "ion 6 12"
    )

    # nozzle box
    box = sim.add_volume("Box", "box")
    box.size = [500 * mm, 500 * mm, 1000 * mm]
    box.rotation = (
        Rotation.from_euler("z", gantry_angle, degrees=True)
        * Rotation.from_euler("x", -90, degrees=True)
    ).as_matrix()
    if gantry_angle == 0:
        box.translation = [0 * mm, -1148 * mm, 0 * mm]  # [1148 *mm, 0 * mm, 0 * mm]
    elif gantry_angle == 90:
        box.translation = [1148 * mm, 0 * mm, 0 * mm]
    box.material = "Vacuum"
    box.color = [0, 0, 1, 1]

    # nozzle WET
    nozzle = sim.add_volume("Box", "nozzle")
    nozzle.mother = box.name
    nozzle.size = [500 * mm, 500 * mm, 2 * mm]
    nozzle.material = "G4_WATER"

    # patient
    target = sim.add_volume("Box", "patient")
    target.size = [252 * mm, 252 * mm, 220 * mm]
    # patient.mother = phantom.name
    # patient.translation = list((img_origin - origin_when_centered) - iso)
    target.material = "G4_WATER"  # material used by default
    # patient.voxel_materials = [
    #     [-1024, -300, "G4_AIR"],
    #     [-300, 3000, "G4_WATER"],
    # ]
    sim.set_max_step_size(target.name, 0.8)

    # physics
    p = sim.get_physics_user_info()
    p.physics_list_name = "FTFP_INCLXX_EMZ"  #'QGSP_BIC_HP_EMZ' #"FTFP_INCLXX_EMZ"
    sim.physics_manager.set_production_cut("world", "all", 1000 * km)

    # add dose actor
    dose_postprocess = sim.add_actor("DoseActor", "dose_postprocess")
    dose_postprocess.output = output_path / "dose_volume.mhd"
    dose_postprocess.mother = target.name
    dose_postprocess.size = [63, 63, 55]
    dose_postprocess.spacing = [4 * mm, 4 * mm, 4 * mm]
    dose_postprocess.hit_type = "random"
    dose_postprocess.dose = True
    dose_postprocess.dose_calc_on_th_fly = (
        False  # calc dose as edep/mass after end of simulation
    )

    dose_in_step = sim.add_actor("DoseActor", "dose_in_step")
    dose_in_step.output = output_path / "dose_volume.mhd"
    dose_in_step.mother = target.name
    dose_in_step.size = [63, 63, 55]
    dose_in_step.spacing = [4 * mm, 4 * mm, 4 * mm]
    dose_in_step.hit_type = "random"
    dose_in_step.dose = True  # calculate dose directly in stepping action
    dose_in_step.dose_calc_on_th_fly = True

    ## source
    nSim = 4000  # 328935  # particles to simulate per beam
    tps = TreatmentPlanSource("RT_plan", sim)
    tps.set_beamline_model(IR2HBL)
    tps.set_particles_to_simulate(nSim)
    tps.set_spots(spots)
    tps.initialize_tpsource()
    actual_n_sim = tps.actual_sim_particles

    # start simulation
    run_simulation = True
    if run_simulation:
        # add stat actor
        s = sim.add_actor("SimulationStatisticsActor", "Stats")
        s.track_types_flag = True
        # start simulation
        output = sim.start()

        # print results at the end
        stat = output.get_actor("Stats")
        print(stat)

    # read output
    img_mhd_out = itk.imread(dose_postprocess.output)
    img_mhd_ref = itk.imread(dose_in_step.output)

    ok = utility.assert_images(
        dose_in_step.output,
        dose_postprocess.output,
        tolerance=10,
    )

    utility.test_ok(ok)
