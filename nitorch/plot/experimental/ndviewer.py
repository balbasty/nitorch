import matplotlib
from matplotlib.widgets import Slider
from matplotlib.image import AxesImage
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import copy
import torch
from ...core import pyutils


class Coordinate:

    def __init__(self, coord=None, dim=None):
        if coord is not None:
            self._coord = pyutils.make_list(coord)
        elif dim is not None:
            self._coord = [0] * dim
        else:
            self._coord = []
        self._cnt = 0
        self._observers = {}

    def __len__(self):
        return len(self._coord)

    def get_coord(self, dim=None):
        """Get current value of the coordinates

        Parameters
        ----------
        dim : int or list[int], optional
            Get coordinate at specified dimensions.

        Returns
        -------
        coord : float or list[float]
            Coordinate(s)

        """
        if dim is not None:
            if isinstance(dim, (list, tuple)):
                return [self.get_coord(d) for d in dim]
            elif dim >= len(self._coord):
                return 0
            else:
                return self._coord[dim]
        else:
            return copy.copy(self._coord)

    def _set_coord(self, coord, dim=None):
        if dim is None:
            self._coord = pyutils.make_list(coord)
        else:
            if isinstance(dim, (list, tuple)):
                for c, d in zip(coord, dim):
                    self._set_coord(c, d)
            else:
                self._coord += [0] * max(0, dim-len(self._coord)+1)
                self._coord[dim] = coord

    def set_coord(self, coord, dim=None):
        """Set value(s) of the coordinates

        Parameters
        ----------
        coord : float or list[float]
            New value(s)
        dim : int or list[int], optional
            Index(es) of the new value(s)

        Returns
        -------
        self : Coordinate

        """
        old_coord = pyutils.make_list(self.get_coord(dim))
        self._set_coord(coord, dim)
        new_coord = pyutils.make_list(self.get_coord(dim))
        # make same length
        length = max(len(new_coord), len(old_coord))
        old_coord = pyutils.make_list(old_coord, length, default=0)
        new_coord = pyutils.make_list(new_coord, length, default=0)
        changed_dim = set(d for d, (old, new) in enumerate(zip(old_coord, new_coord))
                          if old != new)
        self._has_changed(changed_dim)
        return self

    def _has_changed(self, changed_dim):
        """Trigger callbacks"""
        def empty_intersection(a, b):
            a = pyutils.make_set(a)
            b = pyutils.make_set(b)
            return len(set.intersection(a, b)) == 0
        if len(changed_dim) == 0:
            return
        for func, observed_dim in self._observers.values():
            if observed_dim is None \
                    or not empty_intersection(changed_dim, observed_dim):
                func(self.get_coord(observed_dim))

    def on_changed(self, func, dim=None):
        """When the coordinate value (at dimension `dim`) changes,
        call `func` with argument `self.coord` (or `self.coord[dim])

        Parameters
        ----------
        func : callable
            Callback function.
        dim : int or list[int], optional
            Index of watched dimensions.

        Returns
        -------
        id : int
            Identifier of the observer.

        """
        id = self._cnt
        self._observers[id] = (func, dim)
        self._cnt += 1
        return id

    def disconnect(self, id):
        """
        Remove the observer with connection id `id`

        Parameters
        ----------
        id : int
            Connection id of the observer to be removed
        """
        try:
            del self._observers[id]
        except KeyError:
            pass
        return self


class SubCoordinate:

    def __init__(self, coord, dim):
        if not isinstance(coord, Coordinate):
            raise TypeError('Parent coordinate must be a `Coordinate` object')
        if not isinstance(dim, (int, list, tuple)):
            raise TypeError('Sub dimension must be an integer or sequence')

        self._coord = coord
        self._dim = dim
        self._cnt = 0
        self._observers = {}

    def __len__(self):
        if isinstance(self._dim, (list, tuple)):
            return len(self._dim)
        else:
            return 0

    def get_coord(self, dim=None):
        """Get current value of the coordinates

        Parameters
        ----------
        dim : int or list[int], optional
            Get coordinate at specified dimensions.

        Returns
        -------
        coord : float or list[float]
            Coordinate(s)

        """
        if dim is not None:
            if isinstance(dim, (list, tuple)):
                dim = [self._dim[d] for d in dim]
            else:
                dim = self._dim[dim]
        else:
            dim = self._dim
        return self._coord.get_coord(dim)

    def set_coord(self, coord, dim=None):
        """Set value(s) of the coordinates

        Parameters
        ----------
        coord : float or list[float]
            New value(s)
        dim : int or list[int], optional
            Index(es) of the new value(s)

        Returns
        -------
        self : Coordinate

        """
        if dim is not None:
            if isinstance(dim, (list, tuple)):
                dim = [self._dim[d] for d in dim]
            else:
                dim = self._dim[dim]
        else:
            dim = self._dim
        self._coord.set_coord(coord, dim)
        return self

    def on_changed(self, func, dim=None):
        """When the coordinate value (at dimension `dim`) changes,
        call `func` with argument `self.coord` (or `self.coord[dim])

        Parameters
        ----------
        func : callable
            Callback function.
        dim : int or list[int], optional
            Index of watched dimensions.

        Returns
        -------
        id : int
            Identifier of the observer.

        """
        if dim is not None:
            if isinstance(dim, (list, tuple)):
                dim = [self._dim[d] for d in dim]
            else:
                dim = self._dim[dim]
        else:
            dim = self._dim
        return self._coord.on_changed(func, dim)

    def disconnect(self, id):
        """
        Remove the observer with connection id `id`

        Parameters
        ----------
        id : int
            Connection id of the observer to be removed
        """
        try:
            del self._coord.disconnect[id]
        except KeyError:
            pass
        return self


class CoordinateSlider(Slider):

    def __init__(self, ax, coord, label='', valmin=0, valmax=1, valinit=None,
                 **kwargs):
        if valinit is None:
            valinit = (valmax-valmin)/2
        super().__init__(ax, label, valmin, valmax, valinit, **kwargs)
        self._coord = coord
        self._coord.on_changed(self.set_val)
        self.on_changed(self._coord.set_coord)


class AxesImageSection(AxesImage):

    def __init__(self, ax, coord, section=None, rgb=None, **kwargs):
        aspect = kwargs.pop('aspect', None)
        if aspect is None:
            aspect = matplotlib.rcParams['image.aspect']
        ax.set_aspect(aspect)
        super().__init__(ax, **kwargs)

        if section is None:
            section = [0, 1]
        if not isinstance(section, (list, tuple, set)) \
                or len(set(section)) != 2:
            raise TypeError('`section` should be a two-element tuple')
        self._section = tuple(section)
        self._rgb = rgb
        self._coord = coord
        self._coord.on_changed(self.set_section_and_draw)
        self._ndA = []

        if self.get_clip_path() is None:
            # image does not already have clipping set, clip to axes patch
            self.set_clip_path(self.axes.patch)

    def set_data(self, A):
        if torch.is_tensor(A):
            # ensure it can be converted to numpy
            A = A.detach().cpu()
        self._ndA = A
        self.set_section()

    def set_section(self, coord=None):
        if coord is None:
            coord = self._coord.get_coord()
        coord = [int(c) for c in coord]
        self._coord.set_coord(coord)

        # extract section (+ rgb channel)
        coord[self._section[0]] = slice(None)
        coord[self._section[1]] = slice(None)
        if self._rgb is not None:
            coord[self._rgb] = slice(None)
        A = self._ndA.__getitem__(tuple(coord))

        # ensure that the rgb channel is last
        if self._rgb is not None:
            nb_dim = len(self._ndA.shape)
            dims = [nb_dim+d if d < 0 else d for d in [*self._section, self._rgb]]
            dim_rgb = torch.as_tensor(dims).argsort[-1]
            A = torch.as_tensor(A).transpose(dim_rgb, -1)

        # set section as image data
        super().set_data(A)

        self.set_extent(self.get_extent())

    def set_section_and_draw(self, coord=None):
        self.set_section(coord)
        self.axes.figure.canvas.draw()


def show_section(*args, **kwargs):
    # show_section(fig_or_
    import matplotlib.pyplot as plt

    if len(args) == 2:
        fig, data = args
    else:
        data = args[0]
        fig = None
    section = kwargs.get('section', None)
    rgb = kwargs.get('rgb', None)

    if fig is None:
        fig = plt.gcf()

    shape = data.shape
    nb_dim = len(shape)
    nb_sliders = nb_dim - 2 - (1 if rgb is not None else 0)
    grid = GridSpec(nb_sliders + 1, 1, fig,
                    height_ratios=[80] + [20/nb_sliders] * nb_sliders)

    # common coordinate system
    coord = Coordinate(dim=nb_dim)

    # define section panel
    ax_section = fig.add_subplot(grid[0, 0])
    section = AxesImageSection(ax_section, coord, section=section, rgb=rgb)
    section.set_data(data)
    ax_section.add_image(section)
    ax_section.set_xticks([])
    ax_section.set_yticks([])

    # define slider panels
    sliders = []
    for d in range(nb_dim):
        if d in [*section._section, section._rgb]:
            continue
        sub = SubCoordinate(coord, d)
        ax1 = fig.add_subplot(grid[len(sliders)+1, 0])
        label = 'dim: {}'.format(d)
        slider = CoordinateSlider(ax1, sub, label=label,
                                  valmin=0, valmax=shape[d]-1, valstep=1)
        sliders.append(slider)

    fig.show()
    return fig


