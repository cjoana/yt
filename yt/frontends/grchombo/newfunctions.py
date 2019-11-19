
import numpy as np

from yt.units.unit_object import Unit

from yt.fields.derived_field import \
    ValidateSpatial

from yt.geometry.geometry_handler import \
    is_curvilinear

from yt.fields.vector_operations import \
    create_magnitude_field


def setup_partial_derivative_fields(registry, grad_field, field_units, slice_info=None):
    # Current implementation for gradient is not valid for curvilinear geometries
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
            sqrt_det = data[ftype, "chi"]**(-1.5)    # This corrects the ds due to the conformal decomposition
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
        registry.add_field((ftype, "%s_gradient_%s" % (fname, ax)),
                           sampling_type="cell",
                           function=f,
                           validators=[ValidateSpatial(1, [grad_field])],
                           units=grad_units)
    create_magnitude_field(registry, "%s_gradient" % fname,
                           grad_units, ftype=ftype,
                           validators=[ValidateSpatial(1, [grad_field])])


def add_partial_derivative_fields(self, input_field):
    """Add gradient fields.

    Creates four new grid-based fields that represent the components of
    the gradient of an existing field, plus an extra field for the magnitude
    of the gradient. Currently only supported in Cartesian geometries. The
    gradient is computed using second-order centered differences.

    Parameters
    ----------
    input_field : tuple
       The field name tuple of the particle field the deposited field will
       be created from.  This must be a field name tuple so yt can
       appropriately infer the correct field type.

    Returns
    -------
    A list of field name tuples for the newly created fields.

    Examples
    --------
    >>> grad_fields = ds.add_gradient_fields(("chombo","chi"))
    >>> print(grad_fields)
    [('chombo', 'chi_partial_dx'),
     ('chombo', 'chi_partial_dy'),
     ('chombo', 'chi_partial_dz'),
     ('chombo', 'chi_partial_derivative_magnitude')]
    """
    self.index
    if isinstance(input_field, tuple):
        ftype, input_field = input_field[0], input_field[1]
    else:
        raise RuntimeError
    units = self.field_info[ftype, input_field].units
    setup_partial_derivative_fields(self.field_info, (ftype, input_field), units)
    # Now we make a list of the fields that were just made, to check them
    # and to return them
    grad_fields = [(ftype, input_field + "_partial_d%s" % suffix)
                   for suffix in "xyz"]
    grad_fields.append((ftype, input_field + "_partial_derivative_magnitude"))
    deps, _ = self.field_info.check_derived_fields(grad_fields)
    self.field_dependencies.update(deps)
    return grad_fields


def _get_components(self):
    atts = self._handle['/'].attrs
    num_com = self._handle['/'].attrs['num_components']

    return [atts['component_{}'.format(i)] for i in range(int(num_com))]


def _get_domain(self):
    domain = dict()
    domain['N'] = self.domain_dimensions
    domain['L'] = self.domain_width[0]
    domain['L0'] = 0
    domain['dt'] = self._handle['/level_0'].attrs['dt']
    domain['dt_multiplier'] = float(domain['N'][0] / domain['L'] * domain['dt'])

    return domain


def _get_data(self, fields, domain='default', res='default'):


    if domain is 'default':
        domain = _get_domain(self)

    if res is 'default':
        res = domain['N']*1j
    else:
        res = res*1j

    L = domain['L']
    if 'L0' not in domain.keys(): domain['L0'] = 0
    L0 = domain['L0']

    dset = dict()
    reg = self.r[L0:L:res[0], L0:L:res[1], L0:L:res[2]]
    for i, field in enumerate(fields):
        dset[field] = reg[field]

    return dset


def create_dataset_level0(self, component_names,
                          filename, data='default',
                          domain='default',
                          overwrite=False):
    import h5py as h5
    import os

    if domain is 'default':
        domain = _get_domain(self)
    if data is 'default':
        data = _get_data(self, component_names)

    N = int(domain['N'][0])
    L = domain['L']
    dt_multiplier = domain['dt_multiplier']



    """
    Mesh and Other Params
    """
    # def base attributes
    base_attrb = dict()
    base_attrb['time'] = self._handle['/'].attrs['time']
    base_attrb['iteration'] = self._handle['/'].attrs['iteration']
    base_attrb['max_level'] = self._handle['/'].attrs['max_level']
    base_attrb['num_components'] = len(component_names)
    base_attrb['num_levels'] = 1
    base_attrb['regrid_interval_0'] = 1
    base_attrb['steps_since_regrid_0'] = 0
    for comp, name in enumerate(component_names):
        key = 'component_' + str(comp)
        tt = 'S' + str(len(name))
        base_attrb[key] = np.array(name, dtype=tt)

    # def Chombo_global attributes
    chombogloba_attrb = dict()
    chombogloba_attrb['testReal'] = self._handle['/Chombo_global'].attrs['testReal']
    chombogloba_attrb['SpaceDim'] = self._handle['/Chombo_global'].attrs['SpaceDim']

    # def level0 attributes
    level_attrb = dict()
    level_attrb['dt'] = float(L) / N * dt_multiplier
    level_attrb['dx'] = float(L) / N
    level_attrb['time'] = self._handle['/level_0'].attrs['time']
    level_attrb['is_periodic_0'] = self._handle['/level_0'].attrs['is_periodic_0']
    level_attrb['is_periodic_1'] = self._handle['/level_0'].attrs['is_periodic_1']
    level_attrb['is_periodic_2'] = self._handle['/level_0'].attrs['is_periodic_2']
    level_attrb['ref_ratio'] = self._handle['/level_0'].attrs['ref_ratio']
    level_attrb['tag_buffer_size'] = self._handle['/level_0'].attrs['tag_buffer_size']
    prob_dom = (0, 0, 0, N - 1, N - 1, N - 1)
    prob_dt = np.dtype([('lo_i', '<i4'), ('lo_j', '<i4'), ('lo_k', '<i4'),
                        ('hi_i', '<i4'), ('hi_j', '<i4'), ('hi_k', '<i4')])
    level_attrb['prob_domain'] = np.array(prob_dom, dtype=prob_dt)
    boxes = np.array([(0, 0, 0, N - 1, N - 1, N - 1)],
                     dtype=[('lo_i', '<i4'), ('lo_j', '<i4'), ('lo_k', '<i4'), ('hi_i', '<i4'), ('hi_j', '<i4'),
                            ('hi_k', '<i4')])

    """"
    CREATE HDF5
    """

    if overwrite:
        if os.path.exists(filename):
            os.remove(filename)
    else:
        if os.path.exists(filename):
            raise Exception(">> The file already exists, and set not to overwrite.")


    h5file = h5.File(filename, 'w')  # New hdf5 file I want to create

    # base attributes
    for key in base_attrb.keys():
        h5file.attrs[key] = base_attrb[key]

    # group: Chombo_global
    chg = h5file.create_group('Chombo_global')
    for key in chombogloba_attrb.keys():
        chg.attrs[key] = chombogloba_attrb[key]

    # group: levels
    l0 = h5file.create_group('level_0')
    for key in level_attrb.keys():
        l0.attrs[key] = level_attrb[key]
    sl0 = l0.create_group('data_attributes')
    dadt = np.dtype([('intvecti', '<i4'), ('intvectj', '<i4'), ('intvectk', '<i4')])
    sl0.attrs['ghost'] = np.array((3, 3, 3), dtype=dadt)
    sl0.attrs['outputGhost'] = np.array((0, 0, 0), dtype=dadt)
    sl0.attrs['comps'] = base_attrb['num_components']
    sl0.attrs['objectType'] = np.array('FArrayBox', dtype='S9')

    # level datasets
    dataset = np.zeros((base_attrb['num_components'], N, N, N))
    for i, comp in enumerate(component_names):
        if comp in data.keys():
            dataset[i] = data[comp].T
        else:
            raise Exception(">> Component {} not found in the data dictionary".format(comp))
    fdset = []
    for c in range(base_attrb['num_components']):
        fc = dataset[c].T.flatten()
        fdset.extend(fc)
    fdset = np.array(fdset)

    l0.create_dataset("Processors", data=np.array([0]))
    l0.create_dataset("boxes", data=boxes)
    l0.create_dataset("data:offsets=0", data=np.array([0, (base_attrb['num_components']) * N ** 3]))
    l0.create_dataset("data:datatype=0", data=fdset)

    h5file.close()

    return




def create_dict_level0_dust():
    """
    SET PARAMETERS
    """
    path = "./"
    filename = path + "Radiation_Homogeneous.3d.hdf5"  # Name of the new file to create

    EXTENDED = False
    N = 32
    L = 2.17080
    dt_multiplier = 0.01

    omega = 0.33333333
    mass = 0.  # 1 or 0

    def _transform_PhiToChi(x): return np.exp(-2 * x)

    # Set components

    component_names = [  # The order is important: component_0 ... component_(nth-1)
        "chi",

        "h11", "h12", "h13", "h22", "h23", "h33",

        "K",

        "A11", "A12", "A13", "A22", "A23", "A33",

        "Theta",

        "Gamma1", "Gamma2", "Gamma3",

        "lapse",

        "shift1", "shift2", "shift3",

        "B1", "B2", "B3",

        "density", "energy", "pressure", "enthalpy",

        "u0", "u1", "u2", "u3",

        "D", "E", "W",

        "Z0", "Z1", "Z2", "Z3",

        "V1", "V2", "V3",

        "Ham",

        "Ham_ricci", "Ham_trA2", "Ham_K", "Ham_rho",

        "Mom1", "Mom2", "Mom3"
    ]
    temp_comp = np.zeros((N, N, N))  # template for components: array [Nx, Ny. Nz]
    dset = dict()
    # Here set the value of the components (default: to zero)

    dset['chi'] = temp_comp.copy() + 1.
    dset['Ham'] = temp_comp.copy()
    dset['h11'] = temp_comp.copy() + 1.
    dset['h22'] = temp_comp.copy() + 1.
    dset['h33'] = temp_comp.copy() + 1.
    dset['lapse'] = temp_comp.copy() + 1.

    dset['D'] = temp_comp.copy()
    dset['E'] = temp_comp.copy()
    dset['density'] = temp_comp.copy()
    dset['energy'] = temp_comp.copy()
    dset['pressure'] = temp_comp.copy()
    dset['enthalpy'] = temp_comp.copy()
    dset['Z0'] = temp_comp.copy()
    dset['u0'] = temp_comp.copy() + 1.
    dset['W'] = temp_comp.copy() + 1.

    rho_emtensor = temp_comp.copy()

    # ## Constructing variables
    indices = []
    for z in range(N):
        for y in range(N):
            for x in range(N):
                # wvl = 2 * np.pi * 4 / L
                ind = x + y * N + z * N ** 2

                dset['chi'][x][y][z] = 1.
                dset['density'][x][y][z] = 0.1
                dset['energy'][x][y][z] = 3 * (dset['density'][x][y][z]) ** (1 / 3)
                dset['pressure'][x][y][z] = omega * dset['density'][x][y][z] * dset['energy'][x][y][z]
                dset['enthalpy'][x][y][z] = mass + dset['energy'][x][y][z] + omega
                dset['D'][x][y][z] = dset['density'][x][y][z]
                dset['E'][x][y][z] = dset['energy'][x][y][z] * dset['density'][x][y][z]

                rho_emtensor[x][y][z] = dset['density'][x][y][z] * dset['enthalpy'][x][y][z] * dset['u0'][x][y][z] * \
                                        dset['u0'][x][y][z] - dset['pressure'][x][y][z]

                indices.append(ind)


    rho_mean = np.mean(np.hstack(rho_emtensor))
    K_const = - np.sqrt(rho_mean * 24 * np.pi)
    dset['K'] = temp_comp.copy() + K_const

    return dset
