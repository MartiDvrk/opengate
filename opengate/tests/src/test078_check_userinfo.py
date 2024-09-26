#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from opengate.exception import GateDeprecationError
from opengate.tests.utility import (
    get_default_test_paths,
    test_ok,
    print_test,
)
from opengate.utility import g4_units
from opengate.managers import Simulation

if __name__ == "__main__":
    paths = get_default_test_paths(
        __file__,
    )

    sim = Simulation()

    m = g4_units.m
    cm = g4_units.cm
    keV = g4_units.keV
    mm = g4_units.mm
    um = g4_units.um
    Bq = g4_units.Bq

    world = sim.world
    world.size = [3 * m, 3 * m, 3 * m]
    world.material = "G4_AIR"

    waterbox = sim.add_volume("Box", "Waterbox")
    waterbox.size = [40 * cm, 40 * cm, 40 * cm]
    waterbox.translation = [0 * cm, 0 * cm, 25 * cm]
    waterbox.material = "G4_WATER"

    source = sim.add_source("GenericSource", "Default")
    source.particle = "gamma"
    source.energy.mono = 80 * keV
    source.direction.type = "momentum"
    source.direction.momentum = [0, 0, 1]
    source.n = 20

    # test wrong attributes
    is_ok = True
    stats = sim.add_actor("SimulationStatisticsActor", "Stats")
    stats.track_types_flag = True
    try:
        stats.mother = "nothing"
        is_ok = False and is_ok
    except GateDeprecationError as e:
        print("Exception caught: ", e)
        pass
    print_test(
        is_ok, f"Try to set deprecated attribute 'mother', it should raise an exception"
    )
    print()

    # set a WRONG attribute
    stats.TOTO = "nothing"

    # check the number of warnings (before the run)
    print(
        f"(before run) Number of warnings for stats object: {stats.number_of_warnings}"
    )
    b = stats.number_of_warnings == 1
    print_test(
        b, f"Try to set a wrong attribute 'TOTO', it should print a single warning"
    )
    is_ok = is_ok and b

    sim.run(start_new_process=True)

    print(
        f"(after run) Number of warnings for stats object: {stats.number_of_warnings}"
    )
    b = stats.number_of_warnings == 1
    print_test(b, f"No additional warning should be raised")
    is_ok = is_ok and b

    test_ok(is_ok)
