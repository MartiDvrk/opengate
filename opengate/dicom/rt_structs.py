import os
import pydicom


def get_RS_file(dcm_dir, ss_ref_uid):
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
        if dcmtype == "RT Structure Set Storage" and ss_ref_uid == ds.SOPInstanceUID:
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
    return rs_data, rs_path


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
