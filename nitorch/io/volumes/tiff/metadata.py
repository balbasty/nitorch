from xml.etree import ElementTree as etree
from nitorch.core import py


def parse_unit(unit):
    """Parse a unit string

    Parameters
    ----------
    unit : str
        String describing a unit (e.g., 'mm')

    Returns
    -------
    factor : float
        Factor that relates the base unit to the parsed unit.
        E.g., `parse_unit('mm')[0] -> 1e-3`
    baseunit : str
        Physical unit without the prefix.
        E.g., `parse_unit('mm')[1] -> 'm'`

    """
    if unit is None or len(unit) == 0:
        return 1.
    if unit == 'pixel':
        return 1., unit
    unit_type = unit[-1]
    unit_scale = unit[:-1]
    mu1 = '\u00B5'
    mu2 = '\u03BC'
    unit_map = {'Y': 1E24, 'Z': 1E21, 'E': 1E18, 'P': 1E15, 'T': 1E12, 'G': 1E9,
                'M': 1E6, 'k': 1E3, 'h': 1e2, 'da': 1E1, 'd': 1E-1, 'c': 1E-2,
                'm': 1E-3, mu1: 1E-6, mu2: 1E-6, 'u': 1E-6, 'n': 1E-9,
                'p': 1E-12, 'f': 1E-15, 'a': 1E-18, 'z': 1E-21, 'y': 1E-24}
    if len(unit_scale) == 0:
        return 1., unit_type
    else:
        return unit_map[unit_scale], unit_type


def ome_zooms(omexml, series=None):
    """Extract zoom factors (i.e., voxel size) from OME metadata

    This function returns the zoom levels *at the highest resolution*.
    Zooms at subsequent resolution (_in the in-plane direction only_)
    can be obtained by multiplying the zooms at the previous resolution
    by 2. (This assumes that pyramid levels are built using a sliding
    window).

    If more than one series is requested, the returned variables
    are wrapped in a tuple E.g.:
    ```python
    >>> zooms, units, axes = ome_zooms(omexml)
    >>> zooms_series1 = zooms[1]
    >>> zooms_series1
    (10., 10., 5.)
    >>> units_series1 = units[1]
    >>> units_series1
    ('mm', 'mm', 'mm')

    Parameters
    ----------
    omexml : str or bytes
        OME-XML metadata
    series : int or list[int] or None, default=all
        Series ID(s)

    Returns
    -------
    zooms : tuple[float]
    units : tuple[str]
    axes : str

    ```

    """
    if not isinstance(omexml, (str, bytes)) or omexml[-4:] != 'OME>':
        return None, None, None

    # Open XML parser (copied from tifffile)
    try:
        root = etree.fromstring(omexml)
    except etree.ParseError:
        try:
            omexml = omexml.decode(errors='ignore').encode()
            root = etree.fromstring(omexml)
        except Exception:
            return None

    single_series = False
    if series is not None:
        single_series = isinstance(series, int)
        series = py.make_list(series)

    all_zooms = []
    all_units = []
    all_axes = []
    n_image = -1
    for image in root:
        # Any number [0, inf) of `image` elements
        if not image.tag.endswith('Image'):
            continue
        n_image += 1

        if series is not None and n_image not in series:
            all_zooms.append(None)
            all_units.append(None)
            all_axes.append(None)
            continue

        for pixels in image:
            # exactly one `pixels` element per image
            if not pixels.tag.endswith('Pixels'):
                continue

            attr = pixels.attrib
            axes = ''.join(reversed(attr['DimensionOrder']))
            physical_axes = [ax for ax in axes if 'PhysicalSize' + ax in attr]
            zooms = [float(attr['PhysicalSize' + ax]) for ax in physical_axes]
            units = [attr.get('PhysicalSize' + ax + 'Unit', '')
                     for ax in physical_axes]

            all_zooms.append(tuple(zooms))
            all_units.append(tuple(units))
            all_axes.append(''.join(physical_axes))

    # reorder series
    if series is not None:
        all_zooms = [all_zooms[d] for d in series]
        all_units = [all_units[d] for d in series]
        all_axes = [all_axes[d] for d in series]
    if single_series:
        return all_zooms[0], all_units[0], all_axes[0]
    else:
        return tuple(all_zooms), tuple(all_units), tuple(all_axes)
