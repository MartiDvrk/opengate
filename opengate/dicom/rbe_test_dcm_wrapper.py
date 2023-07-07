#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 15:16:20 2023

@author: fava
"""

import os
import opengate as gate
import itk
import numpy as np
import pydicom
import glob


class rbe_test_dcm_wrapper:
    # NOTE: we assume that all dcm files concerning a specific plan are in the sane folder
    # Dicom consistency is checked when creating the correspondent object
    def __init__(self, rp_path, clinical=True):
        self.dcm_dir = os.path.dirname(rp_path)  # directory with all dicom files

        # RT plan as beamset_info object
        print("Get RP file")
        self.rp_path = rp_path
        self.beamset_info = gate.beamset_info(rp_path)
        self.uid = self.beamset_info.uid  # same for all files
        self.beams = self.beamset_info.beams
        self.isocenter = self.beams[0].IsoCenter

        # RT doses: dictionary with dose info for each RD file. One RD for each beam
        print("----------------------------")
        print("Get RD files")
        self.rt_doses = dict()
        rd_files = glob.glob(os.path.dirname(rp_path) + "/RD*")
        for rdp in rd_files:
            rd = pydicom.read_file(rdp)
            dose_sum_type = rd.DoseSummationType
            dose_type = rd.DoseType
            series_descript = rd.SeriesDescription
            if str(dose_sum_type).upper() == "PLAN":
                if str(dose_type).upper() == "PHYSICAL":
                    label = "PLAN"
                elif str(dose_type).upper() == "EFFECTIVE":
                    label = "PLAN_RBE"
            elif str(dose_sum_type).upper() == "EVALUATION":
                label = series_descript.split(":")[1].split()[0]
            else:
                label = "_".join([dose_sum_type, series_descript])

            print(label)
            self.rt_doses[label] = gate.dose_info(rd, rdp)

        # CT
        print("----------------------------")
        print("Get CT files")

        _, ct_files = gate.get_series_filenames(self.dcm_dir)
        self.ct_image = gate.ct_image_from_dicom(ct_files, None)

    def preprocess_ct(self):
        ct_orig = self.ct_image.img

        # dose grid info
        plan_dose = self.rt_doses["PLAN"]

        # crop CT
        bb_ct = gate.bounding_box(img=ct_orig)
        bb_dose = gate.bounding_box(img=plan_dose.image)
        ct_padded = ct_orig
        if not bb_dose in bb_ct:
            print("dose matrix IS NOT contained in original CT! adding dose padding")
            bb_dose_padding = gate.bounding_box(bb=bb_ct)
            bb_dose_padding.merge(bb_dose)
            # ibbmin = np.array(ct_hu_overrides.TransformPhysicalPointToIndex(bb_dose_padding.mincorner))
            # ibbmax = np.array(ct_hu_overrides.TransformPhysicalPointToIndex(bb_dose_padding.maxcorner))+1
            ibbmin, ibbmax = bb_dose_padding.indices_in_image(ct_orig)
            ct_padded = gate.crop_and_pad_image(
                ct_orig, ibbmin, ibbmax, -1000
            )  # "air" padding
        # ibbmin,ibbmax = bb_ct.indices_in_image(ct_padded)
        # ct_cropped = gate.crop_and_pad_image(ct_padded,ibbmin,ibbmax,-100) # "air" padding
        ct_cropped = ct_padded
        # new ct grid
        self.preprocessed_ct = gate.ct_image_from_mhd(ct_cropped)

        return self.preprocessed_ct

    def get_container_size(self):
        ct_bb = gate.bounding_box(img=self.preprocessed_ct.img)
        rot_box_size = 2.0001 * np.max(
            np.abs(
                np.stack(
                    [ct_bb.mincorner - self.isocenter, ct_bb.maxcorner - self.isocenter]
                )
            ),
            axis=0,
        )

        return list(rot_box_size)
