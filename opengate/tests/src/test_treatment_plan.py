import opengate as gate
import itk
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation


def start_simulation(rt_plan_path):
    paths = gate.get_default_test_paths(__file__, "gate_test044_pbs")
    output_path = paths.output / "output_test051_rtp"
    ref_path = paths.output_ref / "test051_ref"

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
    km = gate.g4_units("km")
    cm = gate.g4_units("cm")
    mm = gate.g4_units("mm")
    gcm3 = gate.g4_units("g/cm3")

    # add a material database
    sim.add_material_database(paths.gate_data / "HFMaterials2014.db")

    # treatment info
    treatment = gate.radiation_treatment(rt_plan_path, clinical=False)
    structs = treatment.structures
    beamset = treatment.beamset_info
    doses = treatment.rt_doses
    ct_image = treatment.ct_image
    gantry_angle = float(beamset.beam_angles[0])

    # dose grid info
    plan_dose = doses["PLAN"]
    plan_dose_image = plan_dose.image

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

    # lookup tables
    hu_material = "/home/fava/opengate/opengate/data/Schneider2000MaterialsTable.txt"
    hu_density = "/home/fava/opengate/opengate/data/Schneider2000DensitiesTable.txt"

    # preprocessing
    preprocessed_ct = treatment.preprocess_ct()
    ct_cropped = preprocessed_ct.img
    mass_image = gate.create_mass_image(ct_cropped, hu_density)
    mhd_ct_path = str(ref_path / "absolute_dose_ct.mhd")
    itk.imwrite(ct_cropped, mhd_ct_path)
    img_origin = preprocessed_ct.origin
    origin_when_centered = (
        -(preprocessed_ct.physical_size) / 2.0 + preprocessed_ct.voxel_size / 2.0
    )

    # get transl and rot for correct ct positioning
    iso = np.array(beamset.beams[0].IsoCenter)
    couch_rot = float(beamset.beams[0].PatientSupportAngle)

    # container
    phantom = sim.add_volume("Box", "phantom")
    phantom.size = treatment.get_container_size()
    # phantom.translation = list((img_origin - origin_when_centered) - iso)
    phantom.rotation = Rotation.from_euler("y", -couch_rot, degrees=True).as_matrix()
    phantom.material = "G4_AIR"
    phantom.color = [0, 0, 1, 1]
    print(f"{iso = }")
    print(f"{phantom.translation = }")
    print(f"{couch_rot = }")

    # patient
    patient = sim.add_volume("Image", "patient")
    patient.image = mhd_ct_path
    patient.mother = phantom.name
    patient.translation = list((img_origin - origin_when_centered) - iso)
    patient.material = "G4_AIR"  # material used by default
    # patient.voxel_materials = [
    #     [-1024, -300, "G4_AIR"],
    #     [-300, 3000, "G4_WATER"],
    # ]

    tol = 0.05 * gcm3
    patient.voxel_materials, materials = gate.HounsfieldUnit_to_material(
        tol, hu_material, hu_density
    )

    # physics
    p = sim.get_physics_user_info()
    p.physics_list_name = "FTFP_INCLXX_EMZ"  #'QGSP_BIC_HP_EMZ' #"FTFP_INCLXX_EMZ"
    sim.set_cut("world", "all", 1000 * km)

    # add dose actor
    dose = sim.add_actor("DoseActor", "doseInXYZ")
    dose.output = output_path / "abs_dose_ct.mhd"
    dose.mother = patient.name
    dose.size = list(preprocessed_ct.nvoxels)
    dose.spacing = list(preprocessed_ct.voxel_size)
    dose.hit_type = "random"
    dose.gray = True

    ## source
    nplan = treatment.beamset_info.mswtot
    nSim = 40000  # 328935  # particles to simulate per beam
    tps = gate.TreatmentPlanSource("RT_plan", sim)
    tps.set_beamline_model(IR2HBL)
    tps.set_particles_to_simulate(nSim)
    tps.set_spots_from_rtplan(rt_plan_path)
    tps.initialize_tpsource()

    # start simulation
    run_simulation = True
    if run_simulation:
        # add stat actor
        s = sim.add_actor("SimulationStatisticsActor", "Stats")
        s.track_types_flag = True
        # start simulation
        output = sim.start(True)

        # print results at the end
        stat = output.get_actor("Stats")
        print(stat)

    # rescale dose on planned number of primaries
    dose_path = gate.scale_dose(
        str(dose.output).replace(".mhd", "_dose.mhd"),
        nplan / nSim,
        output_path / "threeDdoseWater.mhd",
    )

    # read output
    img_mhd_out = itk.imread(dose_path)
    img_mhd_out.SetOrigin(
        preprocessed_ct.origin
    )  # dose actor by default has origin in [0,0,0]

    print("--CT--")
    print(preprocessed_ct.nvoxels, preprocessed_ct.voxel_size, preprocessed_ct.origin)
    print("--DOSE OUT--")
    print(dose.size, dose.spacing, img_mhd_out.GetOrigin())
    print("--MASS--")
    print(mass_image.shape[::-1], mass_image.GetSpacing(), mass_image.GetOrigin())
    print("--PLAN DOSE--")
    print(
        plan_dose_image.shape[::-1],
        plan_dose_image.GetSpacing(),
        plan_dose_image.GetOrigin(),
    )

    # rescaledose image on rt dose grid
    dose_resampled = gate.resample_dose(img_mhd_out, mass_image, plan_dose_image)

    # write dicom output
    keys_for_dcm = ["DoseGridScaling"]  # add here other dcm tags you want in your dicom
    rd = plan_dose.dicom_obj
    sub_ds = {k: rd[k] for k in rd.dir() if k in keys_for_dcm}
    dcm_name = os.path.join(output_path, "my_output_dose.dcm")
    gate.mhd_2_dicom_dose(
        dose_resampled, beamset.dicom_obj, "PLAN", dcm_name, ds=sub_ds, phantom=True
    )

    # visualization
    # img_out = itk.GetArrayViewFromImage(dose_resampled)
    img_plan = itk.GetArrayViewFromImage(plan_dose_image)
    # gate.plot2D(np.squeeze(np.sum(img_plan, axis=0)), "Z plan", show=True)
    # gate.plot2D(np.squeeze(np.sum(img_out, axis=0)), "Z sim", show=True)
    # gate.plot2D(np.squeeze(np.sum(img_plan, axis=2)), "X plan", show=True)
    # gate.plot2D(np.squeeze(np.sum(img_out, axis=2)), "X sim", show=True)
    # gate.plot2D(np.squeeze(np.sum(img_plan, axis=1)), "Y plan", show=True)
    # gate.plot2D(np.squeeze(np.sum(img_out, axis=1)), "Y sim", show=True)

    # # 1D
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(25, 10))
    # gate.plot_img_axis(ax, plan_dose_image, "x plan", axis="x")
    # gate.plot_img_axis(ax, plan_dose_image, "y plan", axis="y")
    # gate.plot_img_axis(ax, plan_dose_image, "z plan", axis="z")
    # gate.plot_img_axis(ax, dose_resampled, "x sim", axis="x", linestyle="--")
    # gate.plot_img_axis(ax, dose_resampled, "y sim", axis="y", linestyle="--")
    # gate.plot_img_axis(ax, dose_resampled, "z sim", axis="z", linestyle="--")

    # plt.show()

    # TEST gamma index
    threshold = 15 / 100 * np.amax(img_plan)
    ok, perc_pass = gate.test_gamma_index(
        plan_dose_image,
        dose_resampled,
        pass_rate=0.95,
        dd=4,
        dta=2,
        threshold=threshold,
    )

    gate.test_ok(ok)


if __name__ == "__main__":
    # read data frame
    pklFpath = "/home/fava/Data/01_test_cases/test_pool.pkl"
    df = pd.read_pickle(pklFpath)
    l = (df["GroupID"] == 11) & (df["GroupSubID"] == 3)
    # rt_plan_path = dfhbl.loc[dfhbl["TaskID"] == "Geo_2.3", "RP_fpath"].array[0]
    df = df[l]
    for index, row in df.iterrows():
        rt_plan_path = row["RP_fpath"]
        start_simulation(rt_plan_path)
