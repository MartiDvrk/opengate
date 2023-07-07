import os
import opengate as gate
import itk
import numpy as np


class radiation_treatment:
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
        self.rt_doses = gate.dose_info.get_dose_files(
            self.dcm_dir, self.uid, clinical=clinical
        )

        # RT structures
        print("----------------------------")
        print("Get RS file")
        self.ss_ref_uid = self.beamset_info.structure_set_uid
        self.structures = gate.RT_structs(self.dcm_dir, self.ss_ref_uid)
        self.structures_dcm = self.structures.structure_set

        # CT
        print("----------------------------")
        print("Get CT files")
        self.ctuid = (
            self.structures_dcm.ReferencedFrameOfReferenceSequence[0]
            .RTReferencedStudySequence[0]
            .RTReferencedSeriesSequence[0]
            .SeriesInstanceUID
        )

        _, ct_files = gate.get_series_filenames(self.dcm_dir, self.ctuid)
        self.ct_image = gate.ct_image_from_dicom(ct_files, self.ctuid)

    def preprocess_ct(self, enforce_air_outside_ext=True):
        ct_orig = self.ct_image.img
        ct_array = self.ct_image.array

        # dose grid info
        plan_dose = self.rt_doses["PLAN"]

        # overriding voxels outside external ROI with G4_AIR
        ct_hu_overrides = ct_orig
        if enforce_air_outside_ext:
            ext_roi = gate.region_of_interest(
                ds=self.structures_dcm, roi_id=self.structures.external
            )
            ext_mask = ext_roi.get_mask(ct_orig, corrected=False)
            ext_array = itk.GetArrayViewFromImage(ext_mask) > 0
            ct_array[np.logical_not(ext_array)] = -1000  # hu_air
            ct_hu_overrides = itk.GetImageFromArray(ct_array)
            ct_hu_overrides.CopyInformation(ct_orig)

        # crop CT
        bb_ct = gate.bounding_box(img=ct_hu_overrides)
        bb_dose = gate.bounding_box(img=plan_dose.image)
        ct_padded = ct_hu_overrides
        if not bb_dose in bb_ct:
            print("dose matrix IS NOT contained in original CT! adding dose padding")
            bb_dose_padding = gate.bounding_box(bb=bb_ct)
            bb_dose_padding.merge(bb_dose)
            # ibbmin = np.array(ct_hu_overrides.TransformPhysicalPointToIndex(bb_dose_padding.mincorner))
            # ibbmax = np.array(ct_hu_overrides.TransformPhysicalPointToIndex(bb_dose_padding.maxcorner))+1
            ibbmin, ibbmax = bb_dose_padding.indices_in_image(ct_hu_overrides)
            ct_padded = gate.crop_and_pad_image(
                ct_hu_overrides, ibbmin, ibbmax, -1000
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
