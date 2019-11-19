"""
Various definitions for various other modules and routines



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np

from yt.units.unit_object import Unit

from yt.fields.derived_field import \
    ValidateSpatial

from yt.geometry.geometry_handler import \
    is_curvilinear

from yt.fields.vector_operations import \
    create_magnitude_field

def setup_partial_derivative_fields(registry, grad_field, field_units, slice_info=None):
    # Current implementation is not valid for curvilinear geometries
    if is_curvilinear(registry.ds.geometry): return

    assert (isinstance(grad_field, tuple))
    ftype, fname = grad_field
    if slice_info is None:
        sl_left = slice(None, -2, None)
        sl_right = slice(2, None, None)
        div_fac = 2.0
    else:
        sl_left, sl_right, div_fac = slice_info
    slice_3d = (slice(1, -1), slice(1, -1), slice(1, -1))

    def grad_func(axi, ax):
        slice_3dl = slice_3d[:axi] + (sl_left,) + slice_3d[axi + 1:]
        slice_3dr = slice_3d[:axi] + (sl_right,) + slice_3d[axi + 1:]

        def func(field, data):
            sqrt_det = data[ftype, "chi"] ** (-1.5)  # This corrects the ds due to the conformal decomposition
            ds = sqrt_det * div_fac * data[ftype, "d%s" % ax]
            f = data[grad_field][slice_3dr] / ds[slice_3d]
            f -= data[grad_field][slice_3dl] / ds[slice_3d]
            new_field = np.zeros_like(data[grad_field], dtype=np.float64)
            new_field = data.ds.arr(new_field, f.units)
            new_field[slice_3d] = f
            return new_field

        return func

    field_units = Unit(field_units, registry=registry.ds.unit_registry)
    grad_units = field_units / registry.ds.unit_system["length"]

    for axi, ax in enumerate('xyz'):
        f = grad_func(axi, ax)
        registry.add_field((ftype, "%s_partial_d%s" % (fname, ax)),
                           sampling_type="cell",
                           function=f,
                           validators=[ValidateSpatial(1, [grad_field])],
                           units=grad_units)

    # create_magnitude_field(registry, "%s_partial_magnitude" % fname,
    #                        grad_units, ftype=ftype,
    #                        validators=[ValidateSpatial(1, [grad_field])])



def setup_stats_on_fields(registry, grad_field):
    # Current implementation is not valid for curvilinear geometries
    if is_curvilinear(registry.ds.geometry): return

    assert (isinstance(grad_field, tuple))
    ftype, fname = grad_field

    def weighted_cells(data):
        sqrt_det = data[ftype, "chi"] ** (-1.5)  # This corrects the ds due to the conformal decomposition
        ds = sqrt_det * data[ftype, "dx"]
        total_volume = np.sum(sqrt_det * ds ** 3)

        return sqrt_det * data[grad_field] * ds**3 / total_volume

    w_data = weighted_cells()

    # for axi, ax in enumerate('xyz'):
    #     f = grad_func(axi, ax)
    #     registry.add_field((ftype, "%s_partial_d%s" % (fname, ax)),
    #                        sampling_type="cell",
    #                        function=f,
    #                        validators=[ValidateSpatial(1, [grad_field])],
    #                        units=grad_units)

    # create_magnitude_field(registry, "%s_partial_magnitude" % fname,
    #                        grad_units, ftype=ftype,
    #                        validators=[ValidateSpatial(1, [grad_field])])
