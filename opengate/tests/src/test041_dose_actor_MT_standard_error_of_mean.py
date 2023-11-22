#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import numpy as np
from opengate.tests import utility


def run_sim(n_thr, c4_ref=None, paths=None):
    # create the simulation
    sim = gate.Simulation()

    # main options
    ui = sim.user_info
    ui.g4_verbose = False
    ui.g4_verbose_level = 1
    ui.visu = False
    # ui.random_seed = 123456789
    ui.number_of_threads = n_thr
    Ntotal = 10000 * (30 / n_thr) ** 2
    N_per_trhead = Ntotal / ui.number_of_threads
    # units
    m = gate.g4_units.m
    mm = gate.g4_units.mm
    cm = gate.g4_units.cm
    km = gate.g4_units.km
    nm = gate.g4_units.nm
    MeV = gate.g4_units.MeV
    Bq = gate.g4_units.Bq
    kBq = 1000 * Bq

    # add a material database
    # sim.add_material_database(paths.gate_data / "HFMaterials2014.db")

    #  change world size
    world = sim.world
    world.size = [600 * cm, 500 * cm, 500 * cm]
    # world.material = "Vacuum"

    # waterbox
    phantom = sim.add_volume("Box", "phantom")
    phantom.size = [50 * mm, 100 * mm, 100 * mm]
    phantom.translation = [-25 * mm, 0, 0]
    phantom.material = "G4_WATER"
    phantom.color = [0, 0, 1, 1]

    # physics
    p = sim.get_physics_user_info()
    p.physics_list_name = "QGSP_BIC_EMY"
    # sim.set_cut("world", "all", 1000 * km)
    sim.physics_manager.global_production_cuts.all = 1000 * km

    # default source for tests
    source = sim.add_source("GenericSource", "mysource")
    source.energy.mono = 80 * MeV
    source.particle = "proton"
    source.position.type = "disc"
    source.position.rotation = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    source.position.sigma_x = 2 * mm
    source.position.sigma_y = 2 * mm
    source.position.translation = [0, 0, 0]
    source.direction.type = "momentum"
    source.direction.momentum = [-1, 0, 0]
    source.n = N_per_trhead

    dose_size = [100, 1, 1]
    dose_spacing = [0.5 * mm, 100.0 * mm, 100.0 * mm]
    doseActorName_IDD_singleImage = "IDD_singleImage"
    doseActor = sim.add_actor("DoseActor", doseActorName_IDD_singleImage)
    doseActor.output = paths.output / (
        "test041-" + doseActorName_IDD_singleImage + ".mhd"
    )
    doseActor.mother = phantom.name
    doseActor.size = dose_size
    doseActor.spacing = dose_spacing
    doseActor.hit_type = "random"
    doseActor.dose = False
    doseActor.use_more_RAM = False
    doseActor.ste_of_mean = False
    doseActor.uncertainty = True
    doseActor.square = False

    doseActorName_IDD_NthreadImages = "IDD_NthreadImages"
    doseActor = sim.add_actor("DoseActor", doseActorName_IDD_NthreadImages)
    doseActor.output = paths.output / (
        "test041-" + doseActorName_IDD_NthreadImages + ".mhd"
    )
    doseActor.mother = phantom.name
    doseActor.size = dose_size
    doseActor.spacing = dose_spacing
    doseActor.hit_type = "random"
    doseActor.dose = False
    doseActor.use_more_RAM = True
    doseActor.ste_of_mean = True
    doseActor.uncertainty = False
    doseActor.square = False

    doseActorName_IDD_NthreadImages_unbiased = "IDD_NthreadImages_unbiased"
    doseActor = sim.add_actor("DoseActor", doseActorName_IDD_NthreadImages_unbiased)
    doseActor.output = paths.output / (
        "test041-" + doseActorName_IDD_NthreadImages_unbiased + ".mhd"
    )
    doseActor.mother = phantom.name
    doseActor.size = dose_size
    doseActor.spacing = dose_spacing
    doseActor.hit_type = "random"
    doseActor.dose = False
    doseActor.use_more_RAM = True
    doseActor.ste_of_mean_unbiased = True
    doseActor.uncertainty = False
    doseActor.square = False

    # add stat actor
    s = sim.add_actor("SimulationStatisticsActor", "stats")
    s.track_types_flag = True

    # start simulation
    sim.n = int(N_per_trhead)
    # output = sim.run()
    output = sim.run(start_new_process=True)

    # print results at the end
    stat = sim.output.get_actor("stats")
    print(stat)

    # ----------------------------------------------------------------------------------------------------------------
    # tests

    doseFpath_IDD_singleImage = str(
        sim.output.get_actor(doseActorName_IDD_singleImage).user_info.output
    )
    doseFpath_IDD_NthreadImages = str(
        sim.output.get_actor(doseActorName_IDD_NthreadImages).user_info.output
    )
    doseFpath_IDD_NthreadImages_uncert = str(
        sim.output.get_actor(doseActorName_IDD_NthreadImages).user_info.output
    ).replace(".mhd", "-Uncertainty.mhd")
    doseFpath_IDD_NthreadImages_uncert_unbiased = str(
        sim.output.get_actor(doseActorName_IDD_NthreadImages_unbiased).user_info.output
    ).replace(".mhd", "-Uncertainty.mhd")
    doseFpath_IDD_singleImage_uncert = str(
        sim.output.get_actor(doseActorName_IDD_singleImage).user_info.output
    ).replace(".mhd", "-Uncertainty.mhd")

    unused = utility.assert_images(
        doseFpath_IDD_singleImage,
        doseFpath_IDD_NthreadImages,
        stat,
        tolerance=100,
        ignore_value=0,
        axis="x",
    )
    expected_ratio = 1.00
    gate.exception.warning("Test ratio: dose / dose MT cp image for each trhead")
    is_ok = utility.assert_images_ratio(
        expected_ratio,
        doseFpath_IDD_singleImage,
        doseFpath_IDD_NthreadImages,
        abs_tolerance=0.03,
    )
    gate.exception.warning(
        "Test ratio: uncertainty classic / standard error of mean (of each thread)"
    )
    is_ok = utility.assert_images_ratio(
        expected_ratio,
        doseFpath_IDD_singleImage_uncert,
        doseFpath_IDD_NthreadImages_uncert_unbiased,
        abs_tolerance=0.05,
        fn_to_apply=lambda x: np.mean(x),
    )
    gate.exception.warning(
        "Test ratio: unbiased standard error / biased standard error = c4 corr factor "
    )
    if c4_ref:
        is_ok = utility.assert_images_ratio(
            c4_ref,
            doseFpath_IDD_NthreadImages_uncert_unbiased,
            doseFpath_IDD_NthreadImages_uncert,
            abs_tolerance=0.10,
            fn_to_apply=lambda x: np.mean(x),
        )
    return is_ok


if __name__ == "__main__":
    paths = utility.get_default_test_paths(
        __file__, "gate_test041_dose_actor_dose_to_water"
    )

    is_ok_c4 = []
    is_ok_uncert = []
    n_thrV = [6, 30]
    pass_rates_V = []
    c4_referencesV = {
        2: 0.7978845608,
        3: 0.8862269255,
        4: 0.9213177319,
        5: 0.939985603,
        6: 0.951532862,
        7: 0.959368789,
        8: 0.965030456,
        9: 0.9693107,
        10: 0.972659274,
        16: 0.983284169,
        30: 0.9910802775,
        32: 0.99206349,
        100: 0.997477976,
        1000: 0.999749781,
    }
    for n_thr in n_thrV:
        c4_calc = gate.actors.miscactors.standard_error_c4_correction(n_thr)
        if n_thr in c4_referencesV:
            c4_ref = c4_referencesV[n_thr]
            print(f"{n_thr = }")
        else:
            print(n_thr, "not in c4_referenceV", c4_referencesV)
            raise ValueError()
        is_ok_c4.append(np.abs(c4_calc / c4_ref - 1) < 0.01)
        N_rep = 3
        is_ok_run = np.zeros(N_rep)
        for j in np.arange(0, N_rep):
            is_ok_current = run_sim(n_thr, c4_ref, paths=paths)
            is_ok_run[j] = is_ok_current
        pass_rate = np.sum(is_ok_run) / N_rep
        pass_rates_V.append(pass_rate)
        if pass_rate >= 0.65:
            is_ok_this_N_thread = 1
        else:
            is_ok_this_N_thread = 0
        print(f"{is_ok_this_N_thread =}")
        is_ok_uncert.append(is_ok_this_N_thread)

    print(f"{pass_rates_V =}")
    is_ok = False
    if all(is_ok_c4):
        is_ok = True
    else:
        print("Failed because of incorrect calculation of c4 correction function")
    if all(is_ok_uncert):
        is_ok = is_ok and True
    else:
        is_ok = False
        print("Uncertainties not correctly calculated")
        print(is_ok_uncert)
        print(n_thrV)
    utility.test_ok(is_ok)
