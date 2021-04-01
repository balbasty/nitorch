import torch
from nitorch.core import utils, py
import matplotlib
import matplotlib.colors as colors
import matplotlib.text as text
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.artist as artist


class MenuItemProperties:
    def __init__(self, fontsize=14, fontcolor='black', color='gray', alpha=1.0):
        self.fontsize = fontsize
        self.fontcolor = fontcolor
        self.color = color
        self.alpha = alpha
        self.fontcolor_rgb = colors.to_rgba(fontcolor)[:3]
        self.color_rgb = colors.to_rgba(color)[:3]


class MenuItem(artist.Artist):
    padx = 5
    pady = 5

    def __init__(self, fig, label, props=None, hoverprops=None,
                 on_select=None, zorder=100):
        artist.Artist.__init__(self)

        self.set_figure(fig)
        self.set_zorder(zorder)
        self.labelstr = label
        self.props = props or MenuItemProperties(14, 'black',  'gray')
        self.hoverprops = hoverprops or MenuItemProperties(self.props.fontsize,
                                                           self.props.fontcolor,
                                                           'yellow')
        if self.props.fontsize != self.hoverprops.fontsize:
            raise NotImplementedError(
                'support for different font sizes not implemented')
        self.on_select = on_select

        self.rect = patches.Rectangle((0, 0), 1, 1)
        self.text = text.Text(0, 0, label,
                              figure=self.figure,
                              fontsize=self.props.fontsize,
                              color=self.props.fontcolor)
        self.hover = False
        self.set_hover_props(False)

        fig.canvas.mpl_connect('button_release_event', self.check_select)

    def check_select(self, event):
        if self.get_visible() and event.button == 1:  # LEFT
            over, junk = self.text.contains(event)
            if not over:
                return
            if self.on_select is not None:
                self.on_select(self)

    def set_extent(self, x, y, w, h):
        print(self.text.get_text(), x, y, w, h)
        self.text.set_x(x + self.padx)
        self.text.set_y(y + self.pady)
        self.rect.set_x(x)
        self.rect.set_y(y)
        self.rect.set_width(w + 2*self.padx)
        self.rect.set_height(h + 2*self.pady)
        self.hover = False

    def draw(self, renderer):
        print('draw', self.text.get_text(), self.get_visible())
        if self.get_visible():
            self.rect.draw(renderer)
            self.text.draw(renderer)

    def set_hover_props(self, b):
        if b:
            props = self.hoverprops
        else:
            props = self.props

        self.text.set_color(props.fontcolor)
        self.rect.set(facecolor=props.color, alpha=props.alpha)

    def set_hover(self, event):
        'check the hover status of event and return true if status is changed'
        b, junk = self.text.contains(event)

        changed = (b != self.hover)

        if changed:
            self.set_hover_props(b)

        self.hover = b
        return changed


class Menu(artist.Artist):

    def __init__(self, fig, items, zorder=100):

        artist.Artist.__init__(self)
        fig.suppressComposite = True
        self.set_figure(fig)
        self.items = items
        self.set_zorder(zorder)
        for item in items:
            item.set_zorder(zorder)

        self.item_width = 0
        self.item_height = 0
        for item in self.items:
            bbox = item.text.get_window_extent(fig.canvas.renderer)
            self.item_width = max(self.item_width, bbox.x1 - bbox.x0)
            self.item_height = max(self.item_height, bbox.y1 - bbox.y0)

        n_item = len(self.items)
        self.height = (n_item * self.item_height +
                       2 * sum(item.pady for item in items))
        self.pressed = False
        for item in self.items:
            item.set_visible(False)
        self.figure.canvas.mpl_connect('button_release_event', self.on_click)
        self.figure.canvas.mpl_connect('motion_notify_event', self.on_move)

    def on_click(self, event):
        if event.button != 3 and self.pressed:
            self.pressed = False
            for item in self.items:
                item.set_visible(False)
            self.figure.canvas.draw()
            return
        if event.button == 3:  # RIGHT
            print('right click')
            self.pressed = True
            self.set_position(event.x, event.y)
            for item in self.items:
                item.set_visible(True)
            self.figure.canvas.draw()

    def on_move(self, event):
        if self.pressed:
            for item in self.items:
                if item.set_hover(event):
                    self.figure.canvas.draw()
        return False

    def set_position(self, x, y):
        print('set position')
        for item in self.items:
            left = x
            bottom = y - self.item_height - 2*item.pady
            y -= self.item_height + 2*item.pady

            item.set_extent(left, bottom, self.item_width, self.item_height)

    def draw(self, renderer):
        print('draw', renderer, self.pressed)
        for item in self.items:
            item.draw(renderer)





