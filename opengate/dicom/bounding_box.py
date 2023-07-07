#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


class bounding_box(object):
    """
    Define ranges in which things are in 3D space.
    Maybe this should be more dimensionally flexible, to also handle 2D or N
    dimensional bounding boxes.
    """

    def __init__(self, **kwargs):
        """
        Initialize using *one* of the following kwargs:
        * 'bb': copy from another bounding_box object
        * 'img': obtain bounding box from a 'itk' image
        * 'xyz': initialize bounding box with 6 floats, shaped (x1,y1,z1,x2,y2,z2),
          ((x1,x2),(y1,y2),(z1,z2)) or ((x1,y1,z1),(x2,y2,z2)).
        TODO: maybe 'extent' would be a better name for the "limits" data member.
        TODO: maybe the different kind of constructors should be implemented as static methods instead of with kwargs.
        """
        nkeys = len(kwargs.keys())
        self.limits = np.empty((3, 2))
        if nkeys == 0:
            self.reset()
        elif nkeys > 1:
            raise RuntimeError(
                "too many arguments ({}) to bounding box constructor: {}".format(
                    nkeys, kwargs
                )
            )
        elif "bb" in kwargs:
            bb = kwargs["bb"]
            self.limits = np.copy(bb.limits)
        elif "img" in kwargs:
            img = kwargs["img"]
            if len(img.GetOrigin()) != 3:
                raise ValueError("only 3D bounding boxes/images are supported")
            origin = np.array(img.GetOrigin())
            spacing = np.array(img.GetSpacing())
            dims = np.array(img.GetLargestPossibleRegion().GetSize())
            self.limits[:, 0] = origin - 0.5 * spacing
            self.limits[:, 1] = origin + (dims - 0.5) * spacing
        elif "xyz" in kwargs:
            xyz = np.array(kwargs["xyz"], dtype=float)
            if xyz.shape == (3, 2):
                self.limits = xyz
            elif xyz.shape == (2, 3):
                self.limits = xyz.T
            elif xyz.shape == (6,):
                self.limits = xyz.reshape(3, 2)
            else:
                raise ValueError(
                    "unsupported shape for xyz limits: {}".format(xyz.shape)
                )
            if np.logical_not(self.limits[:, 0] <= self.limits[:, 1]).any():
                raise ValueError(
                    "min should be less or equal max but I got min={} max={}".format(
                        self.limits[:, 0], self.limits[:, 1]
                    )
                )

    def reset(self):
        self.limits[:, 0] = np.inf
        self.limits[:, 1] = -np.inf

    def __repr__(self):
        return "bounding box [[{},{}],[{},{}],[{},{}]]".format(
            *(self.limits.flat[:].tolist())
        )

    @property
    def volume(self):
        if np.isinf(self.limits).any():
            return 0.0
        return np.prod(np.diff(self.limits, axis=1))

    @property
    def empty(self):
        return self.volume == 0.0

    def __eq__(self, rhs):
        if self.empty and rhs.empty:
            return True
        return (self.limits == rhs.limits).all()

    def should_contain(self, point):
        apoint = np.array(point, dtype=float)
        assert len(apoint.shape) == 1
        assert apoint.shape[0] == 3
        self.limits[:, 0] = np.min([self.limits[:, 0], apoint], axis=0)
        self.limits[:, 1] = np.max([self.limits[:, 1], apoint], axis=0)

    def should_contain_all(self, points):
        assert np.array(points).shape[1] == 3
        self.should_contain(np.min(points, axis=0))
        self.should_contain(np.max(points, axis=0))

    @property
    def mincorner(self):
        return self.limits[:, 0]

    @property
    def maxcorner(self):
        return self.limits[:, 1]

    @property
    def center(self):
        return 0.5 * (self.mincorner + self.maxcorner)

    def contains(self, point, inner=False, rtol=0.0, atol=1e-3):
        assert len(point) == 3
        apoint = np.array(point)
        if inner:
            # really inside, NOT at the boundary (within rounding errors)
            gt_lo_lim = np.logical_not(
                np.isclose(self.limits[:, 0], apoint, rtol=rtol, atol=atol)
            ) * (apoint > self.limits[:, 0])
            lt_up_lim = np.logical_not(
                np.isclose(self.limits[:, 1], apoint, rtol=rtol, atol=atol)
            ) * (apoint < self.limits[:, 1])
            return (gt_lo_lim * lt_up_lim).all()
        else:
            # inside, or at the boundary (within rounding errors)
            ge_lo_lim = np.isclose(self.limits[:, 0], apoint, rtol=rtol, atol=atol) | (
                apoint > self.limits[:, 0]
            )
            le_up_lim = np.isclose(self.limits[:, 1], apoint, rtol=rtol, atol=atol) | (
                apoint < self.limits[:, 1]
            )
            return (ge_lo_lim * le_up_lim).all()

    def encloses(self, bb, inner=False, rtol=0.0, atol=1e-3):
        return self.contains(bb.mincorner, inner) and self.contains(bb.maxcorner, inner)

    def __contains__(self, item):
        """
        Support for the 'in' operator

        Works only for other bounding boxes, and for 3D points represented by numpy arrays of shape (3,).
        """
        if type(item) == type(self):
            return self.encloses(item)
        else:
            return self.contains(item)

    def add_margins(self, margins):
        # works both with scalar and vector
        # TODO: allow negative margins, but implement appropriate behavior in case margin is larger than the bb.
        self.limits[:, 0] -= margins
        self.limits[:, 1] += margins

    def merge(self, bb):
        if self.empty:
            self.limits = np.copy(bb.limits)
        elif not bb.empty:
            self.should_contain(bb.mincorner)
            self.should_contain(bb.maxcorner)

    def have_overlap(self, bb):
        # not sure this is correct!
        return (
            (not self.empty)
            and (not bb.empty)
            and (self.limits[:, 0] <= bb.limits[:, 1]).all()
            and (bb.limits[:, 0] <= self.limits[:, 1]).all()
        )

    def intersect(self, bb):
        if not self.have_overlap(bb):
            self.reset()
        else:
            self.limits[:, 0] = np.max([self.mincorner, bb.mincorner], axis=0)
            self.limits[:, 1] = np.min([self.maxcorner, bb.maxcorner], axis=0)

    def indices_in_image(self, img, rtol=0.0, atol=1e-3):
        """
        Determine the voxel indices of the bounding box corners in a given image.

        In case the corners are within rounding margin (given by `eps`) on a
        boundary between voxels in the image, then voxels that would only have
        an infinitesimal intersection with the bounding box are not included.
        The parameters `atol` and `rtol` are used to determine with
        `np.isclose` if BB corners are on a voxel boundary. In case the BB
        corners are outside of the image volume, the indices will be out of the
        range (negative, of larger than image size).

        Returns two numpy arrays of size 3 and dtype int, representing the
        inclusive/exclusive image indices of the lower/upper corners,
        respectively.
        """
        # generically, this is what we want to do:
        bb_img = bounding_box(img=img)
        spacing = np.array(img.GetSpacing())
        ibbmin, devmin = np.divmod(self.mincorner - bb_img.mincorner, spacing)
        ibbmax, devmax = np.divmod(self.maxcorner - bb_img.mincorner, spacing)
        # but if we are "almost exactly" on a voxel boundary we need to be picky on which side to land
        below = np.isclose(devmin, spacing, rtol=rtol, atol=atol)
        above = devmax > atol
        ibbmin[below] += 1  # add one IF on a voxel boundary
        ibbmax[above] += 1  # add one UNLESS on a voxel boundary
        return np.int32(ibbmin), np.int32(ibbmax)

    @property
    def xmin(self):
        return self.limits[0, 0]

    @property
    def xmax(self):
        return self.limits[0, 1]

    @property
    def ymin(self):
        return self.limits[1, 0]

    @property
    def ymax(self):
        return self.limits[1, 1]

    @property
    def zmin(self):
        return self.limits[2, 0]

    @property
    def zmax(self):
        return self.limits[2, 1]
