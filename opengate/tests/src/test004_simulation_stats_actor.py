#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate
import opengate.tests.utility as utility

if __name__ == "__main__":
    paths = utility.get_default_test_paths(
        __file__, "gate_test004_simulation_stats_actor"
    )

    # create the simulation
    sim = gate.Simulation()

    # main options
    sim.g4_verbose = False
    sim.g4_verbose_level = 1
    sim.visu = False
    sim.random_engine = "MersenneTwister"

    # set the world size like in the Gate macro
    m = gate.g4_units.m
    world = sim.world
    world.size = [3 * m, 3 * m, 3 * m]

    # add a simple waterbox volume
    waterbox = sim.add_volume("Box", "Waterbox")
    cm = gate.g4_units.cm
    waterbox.size = [40 * cm, 40 * cm, 40 * cm]
    waterbox.translation = [0 * cm, 0 * cm, 25 * cm]
    waterbox.material = "G4_WATER"

    # default source for tests
    keV = gate.g4_units.keV
    mm = gate.g4_units.mm
    Bq = gate.g4_units.Bq
    source = sim.add_source("GenericSource", "Default")
    source.particle = "gamma"
    source.energy.mono = 80 * keV
    source.direction.type = "momentum"
    source.direction.momentum = [0, 0, 1]
    source.activity = 200000 * Bq

    # add stat actor
    sim.add_actor("SimulationStatisticsActor", "Stats")

    # print before init
    print(sim)
    print("-" * 80)
    print(sim.volume_manager.dump_volumes())
    print(sim.source_manager.dump_sources())
    print(sim.actor_manager.dump_actors())
    print("-" * 80)
    print("Volume types :", sim.volume_manager.dump_volume_types())
    print("Source types :", sim.source_manager.dump_source_types())
    print("Actor types  :", sim.actor_manager.dump_actor_types())

    print("Tree of volumes: ", sim.volume_manager.dump_volume_tree())

    # start simulation
    sim.run()
    print(sim.source_manager.dump_sources())

    stats = sim.output.get_actor("Stats")
    print(stats)

    # gate_test4_simulation_stats_actor
    # Gate mac/main.mac
    stats_ref = utility.read_stat_file(paths.gate / "output" / "stat.txt")
    print("-" * 80)
    is_ok = utility.assert_stats(stats, stats_ref, tolerance=0.03)

    utility.test_ok(is_ok)
