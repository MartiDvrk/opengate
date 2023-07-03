# -----------------------------------------------------------------------------
#   Copyright (C): MedAustron GmbH, ACMIT Gmbh and Medical University Vienna
#   This software is distributed under the terms
#   of the GNU Lesser General  Public Licence (LGPL)
#   See LICENSE for further details
# -----------------------------------------------------------------------------

import pydicom
import itk
import numpy as np
import os


class dose_info(object):
    def __init__(self, rd, rdfp):
        self._rd = rd
        self._rdfp = rdfp
        try:
            # beam dose
            self._beamnr = str(
                self._rd.ReferencedRTPlanSequence[0]
                .ReferencedFractionGroupSequence[0]
                .ReferencedBeamSequence[0]
                .ReferencedBeamNumber
            )
        except:
            # plan dose
            self._beamnr = None
        assert self._rd.pixel_array.shape == (
            int(self._rd.NumberOfFrames),
            int(self._rd.Rows),
            int(self._rd.Columns),
        )

    @property
    def filepath(self):
        return self._rdfp

    @property
    def image(self):
        scaling = float(self._rd.DoseGridScaling)
        img = itk.GetImageFromArray(self._rd.pixel_array * scaling)
        img.SetOrigin(self.origin)
        img.SetSpacing(self.spacing)
        return img

    @property
    def ref_uid(self):
        return str(self._rd.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID)

    @property
    def spacing(self):
        return np.array(
            [float(v) for v in self._rd.PixelSpacing[::-1] + [self._rd.SliceThickness]]
        )

    @property
    def nvoxels(self):
        return np.array(self._rd.pixel_array.shape[::-1])

    @property
    def origin(self):
        return np.array([float(v) for v in self._rd.ImagePositionPatient])

    @property
    def is_physical(self):
        return str(self._rd.DoseType).upper() == "PHYSICAL"

    @property
    def is_effective(self):
        return str(self._rd.DoseType).upper() == "EFFECTIVE"

    @property
    def is_beam_dose(self):
        return self._beamnr is not None

    @property
    def is_plan_dose(self):
        return self._beamnr is None

    @property
    def refd_beam_number(self):
        return self._beamnr

    @property
    def dicom_obj(self):
        return self._rd

    @staticmethod
    def get_dose_files(dirpath, rpuid=None, only_physical=False, clinical=True):
        doses = dict()
        # beam_numbers = [str(beam.BeamNumber) for beam in self._rp.IonBeamSequence]
        print("going to find RD dose files in directory {}".format(dirpath))
        print("for UID={} PLAN".format(rpuid if rpuid else "any/all"))
        for s in os.listdir(dirpath):
            if not s[-4:].lower() == ".dcm":
                print("NOT DICOM (no dcm suffix): {}".format(s))
                continue  # not dicom
            fpath = os.path.join(dirpath, s)
            dcm = pydicom.read_file(fpath)
            if "SOPClassUID" not in dcm:
                continue  # not a RD dose file
            if dcm.SOPClassUID.name != "RT Dose Storage":
                continue  # not a RD dose file

            dose_type = str(dcm.DoseType).upper()
            dose_sum_type = str(dcm.DoseSummationType)

            if clinical:
                # check file integrity
                check_file(dcm)

                drefrtp0 = dcm.ReferencedRTPlanSequence[0]
                if rpuid:
                    uid = str(drefrtp0.ReferencedSOPInstanceUID)
                    if uid != rpuid:
                        print("UID {} != RP UID {}".format(uid, rpuid))
                        continue  # dose file for a different plan

                if dose_sum_type.upper() == "PLAN":
                    # the physical or effective plan dose file (hopefully)
                    label = "PLAN"
                else:
                    # a beam dose file (hopefully)
                    label = str(
                        drefrtp0.ReferencedFractionGroupSequence[0]
                        .ReferencedBeamSequence[0]
                        .ReferencedBeamNumber
                    )
                    print("BEAM DOSE FILE FOR {}".format(label))
            else:
                if dose_sum_type.upper() == "PLAN":
                    # the physical or effective plan dose file (hopefully)
                    label = "PLAN"
                else:
                    label = dose_type

            if dose_type == "EFFECTIVE":
                print("got EFFECTIVE=RBE dose for {}".format(label))
                label += "_RBE"
            elif dose_type == "PHYSICAL":
                print("got PHYSICAL dose for {}".format(label))
            else:
                raise RuntimeError(
                    "unknown dose type {} for {} dose, UID={}".format(
                        dose_type, label, uid
                    )
                )
            if label in doses.keys():
                # oopsie!
                raise RuntimeError(
                    "multiple dose files for beamnr/plan={} and UID={}".format(
                        label, uid
                    )
                )
            if dose_type == "PHYSICAL" or not only_physical:
                doses[label] = dose_info(dcm, fpath)
        # if we arrive here, things are probably fine...
        return doses


def loop_over_tags_level(tags, data, missing_keys):
    for key in tags:
        if key not in data:
            missing_keys.append(key)


def check_file(dcm):
    genericTags = [
        "NumberOfFrames",
        "ReferencedRTPlanSequence",
        "Rows",
        "Columns",
        "DoseGridScaling",
        "PixelSpacing",
        "SliceThickness",
        "ImagePositionPatient",
        "DoseType",
        "SOPClassUID",
        "DoseSummationType",
        "DoseUnits",
    ]

    planSeqTag = [
        "ReferencedSOPInstanceUID"
    ]  # "ReferencedFractionGroupSequence" only if "DoseSummationType" == PLAN
    refBeamTag = "ReferencedBeamNumber"
    missing_keys = []

    # check first layer of the hierarchy
    loop_over_tags_level(genericTags, dcm, missing_keys)

    # check referenced RT Plan seq
    if "ReferencedRTPlanSequence" in dcm:
        # check ROI contour sequence
        loop_over_tags_level(planSeqTag, dcm.ReferencedRTPlanSequence[0], missing_keys)

        if "DoseSummationType" in dcm:
            if dcm.DoseSummationType != "PLAN":
                # check also ReferencedFractionGroupSequence
                if (
                    "ReferencedFractionGroupSequence"
                    not in dcm.ReferencedRTPlanSequence[0]
                ):
                    missing_keys.append("ReferencedFractionGroupSequence")
                elif (
                    refBeamTag
                    not in dcm.ReferencedRTPlanSequence[0]
                    .ReferencedFractionGroupSequence[0]
                    .ReferencedBeamSequence[0]
                ):
                    missing_keys.append(
                        "ReferencedBeamNumber under ReferencedRTPlanSequence/ReferencedFractionGroupSequence/ReferencedBeamSequence"
                    )

    if missing_keys:
        raise ImportError("DICOM RD file not conform. Missing keys: ", missing_keys)
    else:
        print("\033[92mRD file ok \033[0m")


# vim: set et softtabstop=4 sw=4 smartindent:
