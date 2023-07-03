# from impl.beamline_model import beamline_model
# from impl.hlut_conf import hlut_conf
import os
import opengate as gate

# import rt_dose as rd
# import rt_plan as rp
# import rt_structs as rs
# import ct_image as ct


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
        rs_data, rs_path = gate.get_RS_file(self.dcm_dir, self.ss_ref_uid)
        gate.check_RS(rs_data)
        self.structures = rs_data

        # CT
        print("----------------------------")
        print("Get CT files")
        self.ctuid = (
            self.structures.ReferencedFrameOfReferenceSequence[0]
            .RTReferencedStudySequence[0]
            .RTReferencedSeriesSequence[0]
            .SeriesInstanceUID
        )

        _, ct_files = gate.get_series_filenames(self.dcm_dir, self.ctuid)
        self.ct_image = gate.ct_image_from_dicom(ct_files, self.ctuid)
