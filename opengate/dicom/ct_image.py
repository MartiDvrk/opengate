#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#   Copyright (C): MedAustron GmbH, ACMIT Gmbh and Medical University Vienna
#   This software is distributed under the terms
#   of the GNU Lesser General  Public Licence (LGPL)
#   See LICENSE for further details
# -----------------------------------------------------------------------------

import os
import pydicom
import numpy as np
import itk
import logging
import time


class ct_image_base:
    @property
    def meta_data(self):
        slicetimes = [int(s.get("InstanceCreationTime", "0")) for s in self._slices]
        return {
            "Institution Name": str(
                self._slices[0].get("InstitutionName", "anonymized")
            ),
            "Series Instance UID": self._uid,
            "Creation Date": str(
                self._slices[0].get("InstanceCreationDate", "anonymized")
            ),
            "Imaging time": "{}-{}".format(min(slicetimes), max(slicetimes)),
            "NVoxelsXYZ": tuple(self.size.tolist()),
            "NVoxelsTOT": np.prod(self.array.shape),
            "Resolution [mm]": tuple(self.voxel_size.tolist()),
            "Origin [mm]": tuple(self.origin.tolist()),
            "Center [mm]": tuple(
                (self.origin + 0.5 * (self.size - 1) * self.voxel_size).tolist()
            ),
        }

    @property
    def img(self):
        # TODO: should we return a copy or a reference?
        return self._img

    @property
    def nvoxels(self):  # already correct order: [x,y,z]
        # more intuitive name than 'size'
        return np.array(self._img.GetLargestPossibleRegion().GetSize())

    @property
    def size(self):
        return np.array(self._img.GetLargestPossibleRegion().GetSize())

    @property
    def physical_size(self):
        return self.size * np.array(self._img.GetSpacing())

    @property
    def voxel_size(self):
        return np.array(self._img.GetSpacing())

    @property
    def origin(self):
        return np.array(self._img.GetOrigin())

    @property
    def array(self):
        return self._img_array

    @property
    def slices(self):
        return self._slices

    @property
    def uid(self):
        return str(self._slices[0].SeriesInstanceUID)

    def write_to_file(self, mhd):
        assert mhd[-4:].lower() == ".mhd"
        itk.imwrite(self._img, mhd)


class ct_image_from_dicom(ct_image_base):
    def __init__(self, flist, uid):
        print(
            "got {} CT files, first={} last={}".format(len(flist), flist[0], flist[-1])
        )
        self._slices = [pydicom.read_file(f) for f in flist]
        print("got {} CT slices".format(len(self._slices)))

        # check slices integrity
        self._check_dcm_slices()

        # sort slices according to their position along the axis
        self._slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        slice_thicknesses = np.round(
            np.diff([s.ImagePositionPatient[2] for s in self._slices]), decimals=2
        )

        # get spacing -> [sx, sy, sz] average size of all ct slices
        pixel_widths = np.round([s.PixelSpacing[1] for s in self._slices], decimals=2)
        pixel_heights = np.round([s.PixelSpacing[0] for s in self._slices], decimals=2)
        spacing = []
        print("going to obtain voxel spacing")
        for sname, spacings in zip(
            ["pixel width", "pixel height", "slice thickness"],
            [pixel_widths, pixel_heights, slice_thicknesses],
        ):
            if 1 < len(set(spacings)):
                # TO DO: define rounding error tolerance
                print(
                    "The {} seems to suffer from rounding issues (or missing slices): min={} mean={} median={} std={} max={}".format(
                        sname,
                        np.min(spacings),
                        np.mean(spacings),
                        np.median(spacings),
                        np.std(spacings),
                        np.max(spacings),
                    )
                )
            spacing.append(np.mean(spacings))
        print("spacing is ({},{},{})".format(*spacing))

        # get origin
        origin = self._slices[0].ImagePositionPatient[:]
        print("origin is ({},{},{})".format(*origin))

        # TODO: is it possible that some self._slices have a different intercept and slope?
        intercept = np.int16(self._slices[0].RescaleIntercept)
        slope = np.float64(self._slices[0].RescaleSlope)
        print("HU rescale: slope={}, intercept={}".format(slope, intercept))

        # stack 2D slices together to get 3D pixel array
        if slope != 1:
            self._img_array = np.stack([s.pixel_array for s in self._slices]).astype(
                np.int16
            )
            self._img_array = (slope * self._img_array).astype(np.int16) + intercept
        else:
            self._img_array = (
                np.stack([s.pixel_array for s in self._slices]).astype(np.int16)
                + intercept
            )
        print(
            "after HU rescale: min={}, mean={}, median={}, max={}".format(
                np.min(self._img_array),
                np.mean(self._img_array),
                np.median(self._img_array),
                np.max(self._img_array),
            )
        )

        # set image properties
        self._img = itk.GetImageFromArray(self._img_array)
        self._img.SetSpacing(tuple(spacing))
        self._img.SetOrigin(tuple(origin))
        self._uid = uid

    def _check_dcm_slices(self):
        for dcm in self._slices:
            genericTags = [
                "InstanceCreationDate",
                "SeriesInstanceUID",
                "ImagePositionPatient",
                "RescaleIntercept",
                "RescaleSlope",
                "InstanceCreationTime",
                "ImagePositionPatient",
                "PixelSpacing",
            ]
            missing_keys = []
            for key in genericTags:
                if key not in dcm:
                    missing_keys.append(key)
            if missing_keys:
                raise ImportError(
                    "DICOM CT file not conform. Missing keys: ", missing_keys
                )
        print("\033[92mCT files ok \033[0m")


def get_series_filenames(ddir, uid=None):
    dcmseries_reader = itk.GDCMSeriesFileNames.New(Directory=ddir)
    ids = dcmseries_reader.GetSeriesUIDs()
    print("got DICOM {} series IDs".format(len(ids)))
    flist = list()
    if uid:
        if uid in ids:
            try:
                # flist = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(ddir,uid)
                flist = dcmseries_reader.GetFileNames(uid)
                return uid, flist
            except:
                print(
                    "something wrong with series uid={} in directory {}".format(
                        uid, ddir
                    )
                )
                raise
    else:
        ctid = list()
        for suid in ids:
            # flist = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(ddir,suid)
            flist = dcmseries_reader.GetFileNames(suid)
            f0 = pydicom.dcmread(flist[0])
            if not hasattr(f0, "SOPClassUID"):
                print(
                    "weird, file {} has no SOPClassUID".format(
                        os.path.basename(flist[0])
                    )
                )
                continue
            descr = pydicom.uid.UID_dictionary[f0.SOPClassUID][0]
            if descr == "CT Image Storage":
                print("found CT series id {}".format(suid))
                ctid.append(suid)
            else:
                print('not CT: series id {} is a "{}"'.format(suid, descr))
        if len(ctid) > 1:
            raise ValueError(
                "no series UID was given, and I found {} different CT image series: {}".format(
                    len(ctid), ",".join(ctid)
                )
            )
        elif len(ctid) == 1:
            uid = ctid[0]
            # flist = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(ddir,uid)
            flist = dcmseries_reader.GetFileNames(uid)
            return uid, flist
