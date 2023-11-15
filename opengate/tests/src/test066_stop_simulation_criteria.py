#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate
from scipy.spatial.transform import Rotation
from opengate.tests import utility
import itk
import numpy as np


def define_run_timing_intervals(
    n_part_per_core, n_part_check, n_cores, skip_first_n_part=0, n_last_run=1000
):
    sec = gate.g4_units.second
    n_tot_planned = n_part_per_core * n_cores
    if skip_first_n_part == 0:
        run_timing_intervals = []
        start_0 = 0
    else:
        run_timing_intervals = [[0, (skip_first_n_part / n_tot_planned) * sec]]
        start_0 = (skip_first_n_part / n_tot_planned) * sec

    end_last = (n_last_run / n_tot_planned) * sec
    n_runs = round(((n_tot_planned - skip_first_n_part - n_last_run) / n_part_check))
    # print(n_runs)

    # end = start + 1 * sec / n_runs
    end = start_0 + (1 * sec - start_0 - end_last) / n_runs
    start = start_0
    for r in range(n_runs):
        run_timing_intervals.append([start, end])
        start = end
        end += (1 * sec - start_0 - end_last) / n_runs

    run_timing_intervals.append([start, start + end_last])
    # print(run_timing_intervals)

    return run_timing_intervals


def calculate_mean_unc(edep_arr, unc_arr, edep_thresh_rel=0.7):
    edep_max = np.amax(edep_arr)
    mask = edep_arr > edep_max * edep_thresh_rel
    unc_used = unc_arr[mask]
    unc_mean = np.mean(unc_used)

    return unc_mean


if __name__ == "__main__":
    paths = utility.get_default_test_paths(
        __file__, "gate_test029_volume_time_rotation", "test030"
    )

    # check statistical uncertainty every n_check simlated particles
    n_planned = 650000
    n_check = 5000
    n_cores = 10
    # n_runs = round(n_planned*n_cores/(n_check))
    run_timing_intervals = define_run_timing_intervals(
        n_planned, n_check, n_cores, skip_first_n_part=50000
    )

    # goal uncertainty
    unc_goal = 0.05
    thresh_voxel_edep_for_unc_calc = 0.7

    # create the simulation
    sim = gate.Simulation()

    # main options
    ui = sim.user_info
    ui.g4_verbose = False
    ui.visu = False
    ui.random_seed = 983456
    ui.number_of_threads = n_cores

    # units
    m = gate.g4_units.m
    mm = gate.g4_units.mm
    cm = gate.g4_units.cm
    um = gate.g4_units.um
    nm = gate.g4_units.nm
    MeV = gate.g4_units.MeV
    Bq = gate.g4_units.Bq
    sec = gate.g4_units.second

    #  change world size
    world = sim.world
    world.size = [1 * m, 1 * m, 1 * m]

    # add a simple fake volume to test hierarchy
    # translation and rotation like in the Gate macro
    fake = sim.add_volume("Box", "fake")
    fake.size = [40 * cm, 40 * cm, 40 * cm]
    fake.translation = [1 * cm, 2 * cm, 3 * cm]
    fake.material = "G4_AIR"
    fake.color = [1, 0, 1, 1]

    # waterbox
    waterbox = sim.add_volume("Box", "waterbox")
    waterbox.mother = "fake"
    waterbox.size = [20 * cm, 20 * cm, 20 * cm]
    waterbox.translation = [-3 * cm, -2 * cm, -1 * cm]
    waterbox.rotation = Rotation.from_euler("y", -20, degrees=True).as_matrix()
    waterbox.material = "G4_WATER"
    waterbox.color = [0, 0, 1, 1]

    # physics
    sim.set_production_cut("world", "all", 700 * um)

    # default source for tests
    # the source is fixed at the center, only the volume will move
    source = sim.add_source("GenericSource", "mysource")
    source.energy.mono = 150 * MeV
    source.particle = "proton"
    source.position.type = "disc"
    source.position.radius = 5 * mm
    source.direction.type = "momentum"
    source.direction.momentum = [0, 0, 1]
    source.activity = n_planned * Bq  # 1 part/s

    # add dose actor
    dose = sim.add_actor("DoseActor", "dose")
    dose.output = paths.output / "test030-edep.mhd"
    dose.mother = "waterbox"
    dose.size = [99, 99, 99]
    mm = gate.g4_units.mm
    dose.spacing = [2 * mm, 2 * mm, 2 * mm]
    dose.translation = [2 * mm, 3 * mm, -2 * mm]
    dose.uncertainty = False
    dose.ste_of_mean = True
    dose.goal_uncertainty = unc_goal
    dose.thresh_voxel_edep_for_unc_calc = thresh_voxel_edep_for_unc_calc

    # add stat actor
    s = sim.add_actor("SimulationStatisticsActor", "Stats")
    s.track_types_flag = True
    s.output = paths.output / "stats030.txt"

    # motion
    sim.run_timing_intervals = run_timing_intervals

    # start simulation
    output = sim.run()

    # print results at the end
    stat = sim.output.get_actor("Stats")
    print(stat)

    dose = sim.output.get_actor("dose")
    print(dose)

    # test that final mean uncertainty satisfies the goal uncertainty
    test_thresh_rel = 0.01
    d = output.get_actor("dose")

    edep_img = itk.imread(paths.output / d.user_info.output)
    edep_arr = itk.GetArrayViewFromImage(edep_img)
    unc_img = itk.imread(paths.output / d.user_info.output_uncertainty)
    unc_array = itk.GetArrayFromImage(unc_img)

    unc_mean = calculate_mean_unc(
        edep_arr, unc_array, edep_thresh_rel=thresh_voxel_edep_for_unc_calc
    )
    print(f"{unc_goal = }")
    print(f"{unc_mean = }")
    ok = unc_mean < unc_goal and unc_mean > unc_goal - test_thresh_rel

    utility.test_ok(ok)
