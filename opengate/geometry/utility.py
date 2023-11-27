from anytree import RenderTree
import numpy as np
from scipy.spatial.transform import Rotation
from box import Box

import opengate_core as g4
from ..definitions import __world_name__
from ..exception import fatal

"""
http://geant4-userdoc.web.cern.ch/geant4-userdoc/UsersGuides/ForApplicationDeveloper/html/Detector/Geometry/geomSolids.html#constructed-solid-geometry-csg-solids
"""


def box_add_size(box, thickness):
    box.size = [x + thickness for x in box.size]


def cons_add_size(cons, thickness):
    cons.rmax1 += thickness / 2
    cons.rmax2 += thickness / 2
    cons.dz += thickness


def translate_point_to_volume(simulation, volume, top, x):
    """

    Consider the point x in the current volume and return the coordinate of x in the top volume
    (that must be an ancestor).
    Translation only, do not consider rotation.
    """
    while volume.name != top:
        x += volume.translation
        volume = simulation.volume_manager.volumes[volume.mother]
    return x


def render_tree(tree, geometry, world_name):
    """
    Print a tree of volume
    """
    s = ""
    for pre, fill, node in RenderTree(tree[world_name]):
        v = geometry[node.name]
        s += f"{pre}{node.name} {v.type_name} {v.material}\n"

    # remove last break line
    return s[:-1]


"""
A rotation matrix (3x3) can be represented by:
- G4RotationMatrix in module opengate_core
- np.array in module numpy
- Rotation in module scipy.spatial.transform

With scipy and np:
- rot_np = rot_scipy.as_matrix()
- rot_scipy = Rotation.from_matrix(rot_np)

With G4RotationMatrix
- rot_g4 = rot_np_as_g4(rot_np)
- rot_np = rot_g4_as_np(rot_g4)

Also for G4ThreeVector
- v_np = vec_g4_as_np(v_g4)
- v_g4 = vec_np_as_g4(v_np)
"""


def is_rotation_matrix(R):
    """
    https://stackoverflow.com/questions/53808503/how-to-test-if-a-matrix-is-a-rotation-matrix
    """
    # square matrix test
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        return False
    should_be_identity = np.allclose(R.dot(R.T), np.identity(R.shape[0], np.float_))
    should_be_one = np.allclose(np.linalg.det(R), 1)
    return should_be_identity and should_be_one


def vec_np_as_g4(v):
    return g4.G4ThreeVector(v[0], v[1], v[2])


def vec_g4_as_np(v):
    vnp = np.zeros(3)
    vnp[0] = v.x
    vnp[1] = v.y
    vnp[2] = v.z
    return vnp


def rot_np_as_g4(rot):
    if not is_rotation_matrix(rot):
        fatal(f"This matrix is not a rotation matrix (not orthogonal): \n{rot}")
    try:
        r = g4.HepRep3x3(
            rot[0, 0],
            rot[0, 1],
            rot[0, 2],
            rot[1, 0],
            rot[1, 1],
            rot[1, 2],
            rot[2, 0],
            rot[2, 1],
            rot[2, 2],
        )
    except Exception as e:
        s = f"Cannot convert the rotation {rot} to a 3x3 matrix. Exception is: "
        s += str(e)
        fatal(s)
    a = g4.G4RotationMatrix()
    a.set(r)
    return a


def rot_g4_as_np(rot):
    r = np.zeros(shape=(3, 3))
    r[0, 0] = rot.xx()
    r[0, 1] = rot.xy()
    r[0, 2] = rot.xz()
    r[1, 0] = rot.yx()
    r[1, 1] = rot.yy()
    r[1, 2] = rot.yz()
    r[2, 0] = rot.zx()
    r[2, 1] = rot.zy()
    r[2, 2] = rot.zz()
    if not is_rotation_matrix(r):
        fatal(f"The G4 matrix is not a rotation matrix (not orthogonal): \n{rot}")
    return r


def get_g4_translation(translation):
    if isinstance(translation, g4.G4ThreeVector):
        return translation
    else:
        return vec_np_as_g4(translation)


def get_g4_rotation(rotation):
    if isinstance(rotation, g4.G4RotationMatrix):
        return rotation
    else:
        return rot_np_as_g4(rotation)


def get_g4_transform(translation=[0, 0, 0], rotation=Rotation.identity().as_matrix()):
    return g4.G4Transform3D(get_g4_rotation(rotation), get_g4_translation(translation))


def get_translation_from_rotation_with_center(rot, center):
    center = np.array(center)
    t = rot.apply(-center) + center
    # note: apply is the same than rot.as_matrix().dot()
    return t


def get_transform_orbiting(position, axis, angle_deg):
    p = np.array(position)
    rot = Rotation.from_euler(axis, angle_deg, degrees=True)
    t = rot.apply(p)
    return t, rot.as_matrix()


def get_transform_world_to_local(vol_name):
    # cumulated translation and rotation
    ctr = [0, 0, 0]
    crot = Rotation.identity().as_matrix()
    first = True
    while vol_name != __world_name__:
        pv = g4.G4PhysicalVolumeStore.GetInstance().GetVolume(vol_name, False)
        tr = vec_g4_as_np(pv.GetObjectTranslation())
        rot = rot_g4_as_np(pv.GetObjectRotation())
        if first:
            ctr = tr
            crot = rot
            first = False
        else:
            crot = np.matmul(rot, crot)
            ctr = rot.dot(ctr) + tr
        vol_name = pv.GetMotherLogical().GetName()

    return ctr, crot


def get_circular_repetition(
    number_of_repetitions,
    first_translation,
    angular_step_deg="auto_full_circle",
    start_angle_deg=0.0,
    first_rotation=Rotation.identity().as_matrix(),
    axis=(0, 0, 1),
):
    if not is_rotation_matrix(first_rotation):
        fatal(f"Invalid rotation matrix 'first_rotation': {first_rotation}.")

    if angular_step_deg == "auto_full_circle":
        angular_step_deg = 360.0 / number_of_repetitions
    elif angular_step_deg == "auto_half_circle":
        angular_step_deg = 360.0 / number_of_repetitions
    elif not isinstance(angular_step_deg, (int, float)):
        fatal(
            f"The input variable 'angular_step_deg' should be a number (int, float) "
            f"or one of the following terms 'auto_full_circle', 'auto_half_circle'. "
            f"Received: {angular_step_deg} which is of type {type(angular_step_deg).__name__}. "
        )

    translations = []
    rotations = []
    for angle in np.arange(
        start_angle_deg,
        start_angle_deg + number_of_repetitions * np.deg2rad(angular_step_deg),
        np.deg2rad(angular_step_deg),
    ):
        rot = Rotation.from_rotvec(angle * np.array(axis))
        rotations.append(rot.as_matrix().dot(first_rotation))
        translations.append(rot.apply(first_translation))

    return translations, rotations


def get_grid_repetition(size, spacing, start=None, return_lut=False):
    """Generate a list of 3-vectors to be used as 'translation' parameter of a volume.

    size:       3-vector or 3-item list specifying the number of repetitions along the axes x, y, z
    spacing:    3-vector or 3-item list specifying the spacing along x, y, z between the translation vectors.
    start:      Optional 3-vector specifying the first translation vector on the grid.
                If not provided, the grid is centered around (0,0,0).
    """

    if start is None:
        start = [-(s - 1) * sp / 2.0 for s, sp in zip(size, spacing)]
    translations = [
        [
            start[0] + spacing[0] * x,
            start[1] + spacing[1] * y,
            start[2] + spacing[2] * z,
        ]
        for x, y, z in np.ndindex(size[0], size[1], size[2])
    ]

    if return_lut is True:
        lut = dict([(i, tr) for i, tr in enumerate(translations)])
        return translations, lut
    else:
        return translations


def volume_orbiting_transform(axis, start, end, n, initial_t, initial_rot):
    angle = start
    step_angle = (end - start) / n
    translations = []
    rotations = []
    for r in range(n):
        irot = Rotation.from_matrix(initial_rot)
        t, rot = get_transform_orbiting(initial_t, axis, angle)
        rot = Rotation.from_matrix(rot)
        rot = rot * irot
        translations.append(t)
        r = rot.as_matrix()
        rotations.append(r)
        angle += step_angle
    return translations, rotations
