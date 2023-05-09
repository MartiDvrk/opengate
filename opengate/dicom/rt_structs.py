import os
import pydicom


class RT_structs:
    def __init__(self, dcm_dir, ss_ref_uid):
        # ss_ref_uid = self.rp_data.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID
        print(
            "going to try to find the file with structure set with UID '{}'".format(
                ss_ref_uid
            )
        )
        nskip = 0
        ndcmfail = 0
        nwrongtype = 0
        rs_data = None
        rs_path = None
        for s in os.listdir(dcm_dir):
            if s[-4:].lower() != ".dcm":
                nskip += 1
                print("no .dcm suffix: {}".format(s))
                continue
            try:
                # print(s)
                ds = pydicom.dcmread(os.path.join(dcm_dir, s))
                dcmtype = ds.SOPClassUID.name
            except:
                ndcmfail += 1
                continue
            if (
                dcmtype == "RT Structure Set Storage"
                and ss_ref_uid == ds.SOPInstanceUID
            ):
                print("found structure set for CT: {}".format(s))
                rs_data = ds
                rs_path = os.path.join(dcm_dir, s)
                break
            else:
                nwrongtype += 1

        if rs_data is None:
            raise RuntimeError(
                "could not find structure set with UID={}; skipped {} with wrong suffix, got {} with 'dcm' suffix but pydicom could not read it, got {} with wrong class UID and/or instance UID. It could well be that this is a commissioning plan without CT and structure set data.".format(
                    ss_ref_uid, nskip, ndcmfail, nwrongtype
                )
            )
        check_RS(rs_data)
        self.structure_set = rs_data
        self.rs_path = rs_path
        self.roinumbers = []
        self.roinames = []
        self.roicontoursets = []
        self.roitypes = []

        # get ROIs
        self.get_ROIs()
        self.external = self.roinames[
            self.roitypes.index("EXTERNAL")
        ]  # TODO: what if none or more than one external ?

    def get_ROIs(self):
        for i, roi in enumerate(self.structure_set.StructureSetROISequence):
            try:
                # logger.debug("{}. ROI number {}".format(i,roi.ROINumber))
                # logger.debug("{}. ROI name   {}".format(i,roi.ROIName))
                roinumber = str(roi.ROINumber)  # NOTE: roi numbers are *strings*
                roiname = str(roi.ROIName)
                contourset = None
                roitype = None
                if i < len(self.structure_set.ROIContourSequence):
                    # normally this works
                    ci = self.structure_set.ROIContourSequence[i]
                    if str(ci.ReferencedROINumber) == roinumber:
                        contourset = ci
                if not contourset:
                    # logger.debug("(nr={},name={}) looks like this is a messed up structure set...".format(roinumber,roiname))
                    for ci in self.structure_set.ROIContourSequence:
                        if str(ci.ReferencedROINumber) == roinumber:
                            # logger.debug("(nr={},name={}) contour found, phew!".format(roinumber,roiname))
                            contourset = ci
                            break
                if not contourset:
                    pass
                    # logger.warn("ROI nr={} name={} does not have a contour, skipping it".format(roinumber,roiname))
                if i < len(self.structure_set.RTROIObservationsSequence):
                    # normally this works
                    obsi = self.structure_set.RTROIObservationsSequence[i]
                    if str(obsi.ReferencedROINumber) == roinumber:
                        roitype = str(obsi.RTROIInterpretedType)
                if not roitype:
                    # logger.debug("(nr={},name={}) looks like this is a messed up structure set...".format(roinumber,roiname))
                    for obsi in self.structure_set.RTROIObservationsSequence:
                        if str(obsi.ReferencedROINumber) == roinumber:
                            roitype = str(obsi.RTROIInterpretedType)
                            # logger.debug("(nr={},name={}) type={} found, phew!".format(roinumber,roiname,roitype))
                            break
                if not roitype:
                    pass
                    # logger.warn("ROI nr={} name={} does not have a type, skipping it".format(roinumber,roiname))
                if bool(roitype) and bool(contourset):
                    self.roinumbers.append(roinumber)
                    self.roinames.append(roiname)
                    self.roicontoursets.append(contourset)
                    self.roitypes.append(roitype)
            except Exception as e:
                raise RuntimeError(
                    "something went wrong with {}th ROI in the structure set: {}".format(
                        i, e
                    )
                )
                # logger.error("skipping that for now, keep fingers crossed")


def check_RS(dcm):
    data = dcm

    # keys and tags used by IDEAL from RS file
    genericTags = [
        "SOPClassUID",
        "SeriesInstanceUID",
        "StructureSetROISequence",
        "ROIContourSequence",
        "RTROIObservationsSequence",
        "ReferencedFrameOfReferenceSequence",
    ]
    structTags = ["ROIName", "ROINumber"]
    contourTags = ["ReferencedROINumber"]
    observTags = ["ReferencedROINumber", "RTROIInterpretedType"]

    ## --- Verify that all the tags are present and return an error if some are missing --- ##

    missing_keys = []

    # check first layer of the hierarchy
    loop_over_tags_level(genericTags, data, missing_keys)

    if "StructureSetROISequence" in data:
        # check structure set ROI sequence
        loop_over_tags_level(structTags, data.StructureSetROISequence[0], missing_keys)

    if "ROIContourSequence" in data:
        # check ROI contour sequence
        loop_over_tags_level(contourTags, data.ROIContourSequence[0], missing_keys)

    if "RTROIObservationsSequence" in data:
        # check ROI contour sequence
        loop_over_tags_level(
            observTags, data.RTROIObservationsSequence[0], missing_keys
        )

    if missing_keys:
        raise ImportError("DICOM RS file not conform. Missing keys: ", missing_keys)
    else:
        print("\033[92mRS file ok \033[0m")


def loop_over_tags_level(tags, data, missing_keys):
    for key in tags:
        if key not in data:
            missing_keys.append(key)
