# import numbers
import copy
import logging
from typing import Union

import matplotlib.colors
import matplotlib.colorbar
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.widgets import PolygonSelector
# from matplotlib.widgets import Slider
from matplotlib.path import Path as MplPath
import mpl_toolkits.axes_grid1
import numpy as np
from datetime import datetime

from .series import Series

POSITIVE_EPS = 1e-3
logger = logging.getLogger(__name__)


class Viewer(object):
    """Viewer -- a graphical tool to display and interact with Series objects.

    Args:
        images (imagedata.Series or list): Series object or list of Series objects to view.
        fig (Figure): matplotlib.plt.figure if already exist (optional).
        ax (Axes): matplotlib axis if already exist (optional).
        follow (bool): Copy ROI to next tag. Default: False.
        colormap (str): Colour map for display. Default: Greys_r.
        norm (str or matplotlib.colors.Normalize): Normalization method. Either linear/log, or
            the `.Normalize` instance used to scale scalar data to the [0, 1] range before
            mapping to colors using colormap.
        colorbar (bool): Display colorbar with image. Default: None: determine colorbar based
            on colormap and norm.
            range before mapping to colors using colormap.
        window (number): Window width of signal intensities. Default: DICOM Window Width.
        level (number): Window level of signal intensities. Default: DICOM Window Center.
        link (bool): Whether scrolling is linked between displayed objects. Default: False.
        onselect (function): call function when roi changes. Default: None.
            When a polygon is completed or modified after completion,
            the *onselect* function is called and passed idx, tag and a list of the vertices as
            ``(xdata, ydata)`` tuples.

    """

    def __init__(self, images, fig=None, ax=None, follow=False,
                 colormap='Greys_r', norm='linear', colorbar=None, window=None, level=None,
                 link=False, onselect=None):

        _name: str = '{}.{}'.format(__name__, self.__init__.__name__)

        self.fig = fig
        self.ax = ax
        if self.ax is None:
            self.ax = default_layout(fig, len(images))
        self.im = {}
        if isinstance(norm, str):
            if norm == 'linear':
                norm = matplotlib.colors.Normalize
            elif norm == 'log':
                norm = matplotlib.colors.LogNorm
            # elif norm == 'centered':
            #     norm = matplotlib.colors.CenteredNorm
            else:
                raise ValueError('Unknown normalization function: {}'.format(norm))
        if colorbar is None:
            colorbar = (colormap != 'Greys_r' or
                        (norm is not None and norm != matplotlib.colors.Normalize))
        for i, im in enumerate(images):
            self.im[i] = build_info(im, colormap, norm, colorbar, window, level)
        self.follow = follow
        self.link = link
        self.cidenter = None
        self.cidleave = None
        self.cidscroll = None
        self.callback_quit = None
        self.vertices = None  # The polygon vertices, as a dictionary of tags of (x,y)
        self.poly = None
        self.paste_buffer = None
        self.callback_onselect = onselect
        self.viewport = {}
        self.viewport_idx = None
        self.set_default_viewport(self.ax)  # Set wanted viewport
        self.update()  # Update view to achieve wanted viewport

    def __repr__(self):
        return object.__repr__(self)

    def __str__(self):
        return "{" + "{0:s} images".format(len(self.im)) + "}"

    def set_default_viewport(self, axes):
        """View as many Series as there are axes"""

        try:
            if len(axes.shape) == 2:
                rows, columns = axes.shape
            elif len(axes.shape) == 1:
                columns = axes.shape[0]
                rows = 1
            else:
                raise ValueError('Cannot set default viewport')
        except AttributeError:
            rows = columns = 1

        self.rows = rows
        self.columns = columns

        # Setup initial view
        self.viewport_set(0)

    def update(self):
        # For each viewport
        for vp_idx in self.viewport:
            vp = self.viewport[vp_idx]
            if vp is None:
                # Clear ax
                continue
            if vp['next'] != vp['present']:
                # We want to show another image in this viewport
                vp['ax'].cla()
                if vp['next'] in self.im:
                    vp['h'] = self.show(vp['ax'], self.im[vp['next']])
                    vp['present'] = vp['next']
                else:
                    raise IndexError("Series {} should be viewed, but does not exist".format(
                        vp['next']
                    ))
            elif vp['next'] is None:
                vp['ax'].cla()
                vp['ax'].set_axis_off()
            # Update present image in viewport
            try:
                im = self.im[vp['present']]
            except KeyError:
                continue
            if not im['modified']:
                continue
            if im['tag_axis'] is not None:
                # 4D viewer
                vp['h'].set_data(im['im'][im['tag'], im['idx'], ...])
                if im['slider'] is not None:
                    im['slider'].valtext.set_text(pretty_tag_value(im))
            elif im['slice_axis'] is not None:
                # 3D viewer
                vp['h'].set_data(im['im'][im['idx'], ...])
            vp['h'].set_clim(vmin=im['vmin'], vmax=im['vmax'])
            # im['ax'].set_ylabel('Slice {}'.format(self.im['idx']))
            # Lower right text
            if im['lower_right_text'] is not None and im['lower_right_data'] != (im['tag'],):
                fmt = ''
                if im['tag_axis'] is not None:
                    fmt = '{}[{}]: {}'.format(im['input_order'], im['tag'], pretty_tag_value(im))
                im['lower_right_text'].txt.set_text(fmt)
                im['lower_right_data'] = (im['tag'],)
            # Lower left text
            if im['color']:
                fmt = 'SL: {0:d}'
                if im['lower_left_text'] is not None and im['lower_left_data'] != im['idx']:
                    im['lower_left_text'].txt.set_text(fmt.format(im['idx']))
                    im['lower_left_data'] = im['idx']
            else:
                fmt, window, level = pretty_window_level(im)
                if im['lower_left_text'] is not None:
                    if im['lower_left_data'] != (window, level, im['idx']):
                        im['lower_left_text'].txt.set_text(fmt.format(im['idx'], window, level))
                        im['lower_left_data'] = (window, level, im['idx'])
            if self.callback_onselect is not None:
                try:
                    vertices = self.vertices[im['idx']]
                except (KeyError, TypeError):
                    vertices = None
                self.callback_onselect(im['idx'], im['tag'], vertices)
            try:
                vp['ax'].axes.figure.canvas.draw()
            except ValueError:
                pass
            im['modified'] = False

    def show(self, ax, im):
        if im is None:
            return None
        if im['slice_axis'] is None:
            # 2D viewer
            h = ax.imshow(im['im'], cmap=im['colormap'], norm=im['norm'])
        elif im['tag_axis'] is None:
            # 3D viewer
            h = ax.imshow(im['im'][im['idx'], ...], cmap=im['colormap'], norm=im['norm'])
        else:
            # 4D viewer
            h = ax.imshow(im['im'][im['tag'], im['idx'], ...], cmap=im['colormap'],
                          norm=im['norm'])
            # Lower right text
            fmt = '{}[{}]: {}'.format(im['input_order'], im['tag'], pretty_tag_value(im))
            im['lower_right_data'] = (im['tag'],)
            im['lower_right_text'] = AnchoredText(fmt,
                                                  prop=dict(size=6, color='white',
                                                            backgroundcolor='black'),
                                                  frameon=False,
                                                  loc='lower right'
                                                  )
            artist = ax.add_artist(im['lower_right_text'])
            artist.set_visible(im['show_text'])
            im['artists'].append(artist)

        # Update lower left text
        if im['color']:
            fmt = 'SL: {0:d}'
            im['lower_left_data'] = (im['idx'])
            im['lower_left_text'] = AnchoredText(fmt.format(im['idx']),
                                                 prop=dict(size=6, color='white',
                                                           backgroundcolor='black'),
                                                 frameon=False,
                                                 loc='lower left'
                                                 )
        else:
            fmt, window, level = pretty_window_level(im)
            im['lower_left_data'] = (window, level, im['idx'])
            im['lower_left_text'] = AnchoredText(fmt.format(im['idx'], window, level),
                                                 prop=dict(size=6, color='white',
                                                           backgroundcolor='black'),
                                                 frameon=False,
                                                 loc='lower left'
                                                 )
        artist = ax.add_artist(im['lower_left_text'])
        artist.set_visible(im['show_text'])
        im['artists'].append(artist)

        # Update upper left text
        fmt = self.upper_left_text(im['im'])
        im['upper_left_text'] = AnchoredText(fmt,
                                             prop=dict(size=6, color='white',
                                                       backgroundcolor='black'),
                                             frameon=False,
                                             loc='upper left'
                                             )
        artist = ax.add_artist(im['upper_left_text'])
        artist.set_visible(im['show_text'])
        im['artists'].append(artist)

        # Update upper right text
        fmt = self.upper_right_text(im['im'])
        im['upper_right_text'] = AnchoredText(fmt,
                                              prop=dict(size=6, color='white',
                                                        backgroundcolor='black'),
                                              frameon=False,
                                              loc='upper right'
                                              )
        artist = ax.add_artist(im['upper_right_text'])
        artist.set_visible(im['show_text'])
        im['artists'].append(artist)
        im['modified'] = True

        if im['colorbar']:
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            # Create fake pcolormesh to create colorbar matching im['colorbar'] and im['norm']
            _ = plt.colorbar(
                cax.pcolormesh(
                    np.array([[im['norm'].vmin, im['norm'].vmax]]),
                    visible=False,
                    cmap=im['colorbar'],
                    vmin=im['norm'].vmin,
                    vmax=im['norm'].vmax
                ),
                label=im['colormap_label'],
                cax=cax
            )

        ax.set_axis_off()
        # if im['slices'] == im2['slices']:
        #    plt.subplots_adjust(bottom=0.1)
        #    self.rax = plt.axes([0.0, 0.0, 0.2, 0.1], frame_on=False)
        #    self.linkbutton = CheckButtons(self.rax, ['Link'], [link])
        #    self.linkclicked = self.linkbutton.on_clicked(self.toggle_button)
        return h

    def pretty_datetime(self, my_date, my_time):
        _name: str = '{}.{}'.format(__name__, self.pretty_datetime.__name__)

        _date = _time = None
        if my_date is not None:
            try:
                _date = datetime.strptime(my_date, '%Y%m%d')
            except ValueError:
                pass
        if my_time is not None:
            try:
                _time = datetime.strptime(my_time, '%H%M%S.%f')
            except ValueError:
                try:
                    _time = datetime.strptime(my_time, '%H%M%S')
                except ValueError:
                    pass
        _date_fmt = ''
        if _date is not None or _time is not None:
            if _date is not None:
                _date_fmt = '{} '.format(_date.strftime("%Y-%m-%d"))
            else:
                logger.debug('{}: Cannot add date for \"{}\"'.format(_name, my_date))
            if _time is not None:
                _date_fmt += _time.strftime("%H:%M:%S")
            else:
                logger.debug('{}: Cannot add time for \"{}\"'.format(_name, my_time))
        return _date_fmt

    def upper_left_text(self, im):
        """Update upper left text
        """

        data = {}
        for attr in ['patientName', 'patientID']:
            try:
                data[attr] = getattr(im, attr, '')
            except ValueError:
                data[attr] = ''
        for attr in ['StudyDate', 'StudyTime']:
            try:
                data[attr] = im.getDicomAttribute(attr)
            except Exception:
                data[attr] = ''
        _date_fmt = self.pretty_datetime(data['StudyDate'], data['StudyTime'])

        fmt = ''
        if data['patientName']:
            pat_nam: str = '{}'.format(data['patientName'])
            while pat_nam[-1] == '^':
                pat_nam = pat_nam[:-1]
            pat_nam = pat_nam.replace('^', ', ')
            fmt = pat_nam
        if data['patientID']:
            fmt += '\n{}'.format(data['patientID'])
        if len(_date_fmt) > 0:
            fmt += '\n{}'.format(_date_fmt)
        return fmt

    def upper_right_text(self, im):
        """Update upper right text
        """

        data = {}
        for attr in ['seriesNumber', 'seriesDescription']:
            try:
                data[attr] = getattr(im, attr, '')
            except ValueError:
                data[attr] = ''
        for attr in ['SeriesDate', 'SeriesTime']:
            try:
                data[attr] = im.getDicomAttribute(attr)
            except Exception:
                data[attr] = ''
        _date_fmt = self.pretty_datetime(data['SeriesDate'], data['SeriesTime'])

        fmt = ''
        if data['seriesNumber']:
            fmt = '{}. '.format(data['seriesNumber'])
        fmt += data['seriesDescription']
        if len(_date_fmt) > 0:
            fmt += '\n{}'.format(_date_fmt)
        return fmt

    def connect_draw(self, roi=None, color='w', callback_quit=None):
        self.poly_color = color
        self.callback_quit = callback_quit
        idx = self.im[0]['idx']
        if roi is None:
            self.poly = {}
            self.vertices = {}
            if self.follow:
                self.poly[0, idx] = MyPolygonSelector(self.ax[0, 0], self.onselect,
                                                      lineprops={'color': self.poly_color},
                                                      tag=(0, idx), copy=self.on_copy,
                                                      paste=self.on_paste)
            else:
                self.poly[idx] = MyPolygonSelector(self.ax[0, 0], self.onselect,
                                                   lineprops={'color': self.poly_color},
                                                   tag=idx, copy=self.on_copy, paste=self.on_paste)
        else:
            self.poly = {}
            self.vertices = roi
            if self.follow:
                for tag in range(self.im[0]['tags']):
                    for i in range(self.im[0]['slices']):
                        vertices = copy.copy(self.vertices[tag, i]) \
                            if (tag, i) in self.vertices else None
                        self.poly[tag, i] = MyPolygonSelector(self.ax[0, 0], self.onselect,
                                                              lineprops={
                                                                  'color': self.poly_color},
                                                              vertices=vertices,
                                                              tag=(tag, i),
                                                              copy=self.on_copy,
                                                              paste=self.on_paste)
                        # Polygon on single slice and tag 0, only
                        if i == idx and tag == 0:
                            assert self.poly[tag, i].tag == (tag, i), \
                                "Tag index mismatch {}!={}".format((tag, i), self.poly[tag, i].tag)
                            self.poly[tag, i].connect_default_events()
                            self.poly[tag, i].set_visible(True)
                            self.poly[tag, i].update()
                        else:
                            assert self.poly[tag, i].tag == (tag, i), \
                                "Tag index mismatch {}!={}".format((tag, i), self.poly[tag, i].tag)
                            self.poly[tag, i].disconnect_events()
                            self.poly[tag, i].set_visible(False)
                            self.poly[tag, i].update()
            else:
                for i in range(self.im[0]['slices']):
                    vertices = copy.copy(self.vertices[i]) if i in self.vertices else None
                    self.poly[i] = MyPolygonSelector(self.ax[0, 0], self.onselect,
                                                     lineprops={'color': self.poly_color},
                                                     vertices=vertices, tag=i, copy=self.on_copy,
                                                     paste=self.on_paste)
                    # Polygon on single slice only
                    if i != idx:
                        assert self.poly[i].tag == i, \
                            "Tag index mismatch {}!={}".format(i, self.poly[i].tag)
                        self.poly[i].disconnect_events()
                        self.poly[i].set_visible(False)
                        self.poly[i].update()
        self.cidscroll = self.fig.canvas.mpl_connect('scroll_event', self.scroll)
        self.cidkeypress = self.fig.canvas.mpl_connect('key_press_event', self.key_press)

    def disconnect_draw(self):
        if self.follow:
            for t in range(self.im[0]['tags']):
                for idx in range(self.im[0]['slices']):
                    if (t, idx) in self.poly and self.poly[t, idx] is not None:
                        self.poly[t, idx].disconnect_events()
        else:
            for idx in range(self.im[0]['slices']):
                if idx in self.poly and self.poly[idx] is not None:
                    self.poly[idx].disconnect_events()
        self.fig.canvas.mpl_disconnect(self.scroll)
        self.fig.canvas.mpl_disconnect(self.cidkeypress)

    def connect(self):
        # Connect to all the events we need
        # self.cidenter = self.fig.canvas.mpl_connect('axes_enter_event', self.enter_axes)
        # self.cidleave = self.fig.canvas.mpl_connect('axes_leave_event', self.leave_axes)
        self.cidscroll = self.fig.canvas.mpl_connect('scroll_event', self.scroll)
        self.cidkeypress = self.fig.canvas.mpl_connect('key_press_event', self.key_press)
        self.cidpress = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.scroll)
        self.fig.canvas.mpl_disconnect(self.cidkeypress)
        self.fig.canvas.mpl_disconnect(self.cidpress)
        self.fig.canvas.mpl_disconnect(self.cidrelease)
        self.fig.canvas.mpl_disconnect(self.cidmotion)

    def onselect(self, vertices):
        idx = self.im[0]['idx']
        tag = None
        if self.follow:
            tag = self.im[0]['tag']
            self.vertices[tag, idx] = copy.copy(vertices)
        else:
            self.vertices[idx] = copy.copy(vertices)
        if self.callback_onselect is not None:
            self.callback_onselect(idx, tag, vertices)

    def on_copy(self, polygon):
        self.paste_buffer = polygon

    def on_paste(self):
        return self.paste_buffer

    # def grid_from_roi(self):
    #     """Return drawn ROI as grid.
    #
    #     Returns:
    #         Numpy ndarray with shape (nz,ny,nx) from original image, dtype ubyte.
    #         Voxels inside ROI is 1, 0 outside.
    #     """
    #     nt, nz, ny, nx = self.im[0]['tags'], self.im[0]['slices'], self.im[0]['rows'],
    #         self.im[0]['columns']
    #     if self.follow:
    #         grid = np.zeros((nt, nz, ny, nx), dtype=np.ubyte)
    #         for idx in range(nz):
    #             last_used_tag = None
    #             for t in range(nt):
    #                 tag = t, idx
    #                 if tag not in self.vertices or self.vertices[tag] is None:
    #                     if last_used_tag is None:
    #                         # Most probably a slice with no ROIs
    #                         continue
    #                     # Propagate last drawn ROI to unfilled tags
    #                     self.vertices[tag] = self.vertices[last_used_tag]
    #                 else:
    #                     last_used_tag = tag
    #                 path = MplPath(self.vertices[tag])
    #                 x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    #                 x, y = x.flatten(), y.flatten()
    #                 points = np.vstack((x, y)).T
    #                 grid[t, idx] = path.contains_points(points).reshape((ny, nx))
    #     else:
    #         grid = np.zeros((nz, ny, nx), dtype=np.ubyte)
    #         for idx in range(nz):
    #             if idx not in self.vertices or self.vertices[idx] is None:
    #                 continue
    #             path = MplPath(self.vertices[idx])
    #             x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    #             x, y = x.flatten(), y.flatten()
    #             points = np.vstack((x, y)).T
    #             grid[idx] = path.contains_points(points).reshape((ny, nx))
    #     return grid

    def get_roi(self):
        """Return drawn ROI.

        Returns:
            Dict of slices, index as [tag,slice] or [slice], each is list of (x,y) pairs.
        """
        vertices = {}
        for tag in self.poly.keys():
            if len(self.poly[tag].verts) > 0:
                vertices[tag] = self.poly[tag].verts
        return vertices

    # def enter_axes(self, event):
    #    if event.inaxes == self.im['ax']:
    #        print('enter_axes', self.im['ax'], event.inaxes)
    #    elif self.im2 is not None and event.inaxes == self.im2['ax']:
    #        print('enter_axes2', self.im2['ax'], event.inaxes)

    # def leave_axes(self, event):
    #    print('leave_axes', event.inaxes)

    def key_press(self, event):
        if event.key == 'up':
            self.scroll_data(event.inaxes, 1)
        elif event.key == 'down':
            self.scroll_data(event.inaxes, -1)
        elif event.key == 'left':
            self.advance_data(event.inaxes, -1)
        elif event.key == 'right':
            self.advance_data(event.inaxes, 1)
        elif event.key == 'pageup':
            self.viewport_advance(-self.rows * self.columns)
        elif event.key == 'pagedown':
            self.viewport_advance(self.rows * self.columns)
        elif event.key == 'ctrl+home':
            self.viewport_set(0)
        elif event.key == 'ctrl+end':
            self.viewport_set(len(self.im) - self.rows * self.columns)
        elif event.key == 'ctrl+left':
            self.viewport_advance(-1)
        elif event.key == 'ctrl+right':
            self.viewport_advance(1)
        elif event.key == 'ctrl+up':
            self.viewport_advance(-self.columns)
        elif event.key == 'ctrl+down':
            self.viewport_advance(self.columns)
        elif event.key == 'H' or event.key == 'h':
            # Hide display
            self.toggle_hide(event.inaxes)
        elif event.key == 'W' or event.key == 'w':
            # Normalize window center/width using a probability histogram
            self.normalize_window(event.inaxes)
        elif event.key == 'Q' or event.key == 'q':
            # Quit Viewer
            # Set present window/level on Series objects
            for i in self.im.keys():
                self.im[i]['im'].windowCenter = self.im[i]['level']
                self.im[i]['im'].windowWidth = self.im[i]['window']
            if self.callback_quit is not None:
                self.callback_quit()
        # else:
        #    print('key_press: {}'.format(event.key))

    def scroll(self, event):
        if event.button == 'up':
            self.scroll_data(event.inaxes, 1)
        elif event.button == 'down':
            self.scroll_data(event.inaxes, -1)

    def find_viewport_from_event(self, inaxes):
        for vp_idx in self.viewport:
            vp = self.viewport[vp_idx]
            if vp is not None:
                ax = vp['ax']
                if inaxes == ax:
                    return vp_idx
        # Do nothing when the event does not match with any viewport axes
        return None

    def find_image_from_event(self, inaxes):
        for vp_idx in self.viewport:
            vp = self.viewport[vp_idx]
            if vp is not None:
                ax = vp['ax']
                if inaxes == ax:
                    im_idx = vp['present']
                    return self.im[im_idx]
        # Do nothing when the event does not match with any viewport axes
        return None

    def scroll_data(self, inaxes, increment):
        im = self.find_image_from_event(inaxes)
        if im is None:
            return
        old_idx = im['idx']
        im['idx'] = min(max(im['idx'] + increment, 0), im['slices'] - 1)
        if self.link:
            # Scroll all images to same index (if possible)
            self.scroll_all_data(im, im['idx'])
        im['modified'] = old_idx != im['idx']
        if self.poly is not None:
            new_idx = im['idx']
            if self.follow:
                old_idx = im['tag'], old_idx
                new_idx = im['tag'], new_idx
            if old_idx in self.poly:
                assert self.poly[old_idx].tag == old_idx, \
                    "Tag index mismatch {}!={}".format(old_idx, self.poly[old_idx].tag)
            if new_idx in self.poly:
                assert self.poly[new_idx].tag == new_idx, \
                    "Tag index mismatch {}!={}".format(new_idx, self.poly[new_idx].tag)
            if im['modified']:
                self.poly[old_idx].disconnect_events()
                self.poly[old_idx].set_visible(False)
                self.poly[old_idx].update()
            if new_idx in self.poly and self.poly[new_idx] is not None:
                self.poly[new_idx].connect_default_events()
                self.poly[new_idx].set_visible(True)
                self.poly[new_idx].update()
            else:
                self.poly[new_idx] = MyPolygonSelector(self.ax[0, 0], self.onselect,
                                                       lineprops={'color': self.poly_color},
                                                       tag=new_idx, copy=self.on_copy,
                                                       paste=self.on_paste)
        # if self.link and self.im['scrollable'] and self.im2['scrollable']:
        #    self.im['idx'] = min(max(self.im['idx'] + increment, 0), self.im['slices']-1)
        #    self.im2['idx'] = self.im['idx']
        # elif inaxes == self.im['ax'] and self.im['scrollable']:
        #    self.im['idx'] = min(max(self.im['idx'] + increment, 0), self.im['slices']-1)
        # elif self.im2 is not None and inaxes == self.im2['ax'] and self.im2['scrollable']:
        #    self.im2['idx'] = min(max(self.im2['idx'] + increment, 0), self.im2['slices']-1)
        self.update()

    def scroll_all_data(self, im, idx):
        for vp_idx in self.viewport:
            vp = self.viewport[vp_idx]
            if vp is not None:
                im_idx = vp['present']
                if im_idx != idx:
                    im2 = self.im[im_idx]
                    old_idx = im2['idx']
                    im2['idx'] = min(max(idx, 0), im2['slices'] - 1)
                    im2['modified'] = old_idx != im2['idx']

    def advance_data(self, inaxes, increment):
        """Advance display to next/previous tag value"""
        im = self.find_image_from_event(inaxes)
        if im is None or im['tag_axis'] is None:
            return
        old_tag = im['tag']
        im['tag'] = min(max(im['tag'] + increment, 0), len(im['tag_axis']) - 1)
        im['modified'] = old_tag != im['tag']
        if self.poly is not None and self.follow and im['modified']:
            new_tag = im['tag']
            idx = im['idx']
            assert self.poly[old_tag, idx].tag == (old_tag, idx), \
                "Tag index mismatch {}!={}".format((old_tag, idx), self.poly[old_tag, idx].tag)
            if (new_tag, idx) not in self.poly and (old_tag, idx) in self.poly and \
                    self.poly[old_tag, idx] is not None:
                # Copy the polygon to next tag when there is none
                self.poly[new_tag, idx] = MyPolygonSelector(self.ax[0, 0], self.onselect,
                                                            lineprops={'color': self.poly_color},
                                                            vertices=self.poly[old_tag, idx].verts,
                                                            tag=(new_tag, idx), copy=self.on_copy,
                                                            paste=self.on_paste)
            assert self.poly[old_tag, idx].tag == (old_tag, idx), \
                "Tag index mismatch {}!={}".format((old_tag, idx), self.poly[old_tag, idx].tag)
            self.poly[old_tag, idx].disconnect_events()
            self.poly[old_tag, idx].set_visible(False)
            self.poly[old_tag, idx].update()
            assert self.poly[new_tag, idx].tag == (new_tag, idx), \
                "Tag index mismatch {}!={}".format((new_tag, idx), self.poly[new_tag, idx].tag)
            self.poly[new_tag, idx].connect_default_events()
            self.poly[new_tag, idx].set_visible(True)
            self.poly[new_tag, idx].update()
        # if self.link and self.im['scrollable'] and self.im2['scrollable']:
        #    self.im['idx'] = min(max(self.im['idx'] + increment, 0), self.im['slices']-1)
        #    self.im2['idx'] = self.im['idx']
        # elif inaxes == self.im['ax'] and self.im['scrollable']:
        #    self.im['idx'] = min(max(self.im['idx'] + increment, 0), self.im['slices']-1)
        # elif self.im2 is not None and inaxes == self.im2['ax'] and self.im2['scrollable']:
        #    self.im2['idx'] = min(max(self.im2['idx'] + increment, 0), self.im2['slices']-1)
        self.update()

    def viewport_advance(self, increment):
        """Advance viewport by given increment
        """

        self.viewport_set(self.viewport_idx + increment)

    def viewport_set(self, position):
        """Set viewport to image position
        """

        images = len(self.im)
        # Position must be in range 0:images-(rows*columns)
        vp_idx = min(position, images - self.rows * self.columns)
        vp_idx = max(vp_idx, 0)
        if vp_idx == self.viewport_idx:
            # No change
            return
        # print('viewport_set: old idx {}, new idx {}'.format(self.viewport_idx, position))
        self.viewport_idx = vp_idx
        new_viewport = {}
        for row in range(self.rows):
            for column in range(self.columns):
                if vp_idx in self.im:
                    new_viewport[vp_idx] = {
                        'ax': self.ax[row, column],
                        'present': None,
                        'next': vp_idx,
                        'h': None
                    }
                else:
                    new_viewport[vp_idx] = None
                    self.ax[row, column].set_axis_off()
                vp_idx += 1
        self.viewport = new_viewport
        self.update()

    def toggle_hide(self, inaxes):
        """Toggle the display of text on images
        """

        # im = self.find_image_from_event(inaxes)
        # if im is None:
        #     return
        for im in self.im:
            self.im[im]['show_text'] = not self.im[im]['show_text']
            for artist in self.im[im]['artists']:
                artist.set_visible(self.im[im]['show_text'])
            self.im[im]['modified'] = True
        self.update()

    def normalize_window(self, inaxes):
        im = self.find_image_from_event(inaxes)
        if im is None:
            return
        # Normalize on displayed slice only
        probs = (0.01, 0.99)
        if im['slice_axis'] is None:
            # 2D data
            vmin, vmax = im['im'].calculate_clip_range(probs)
        elif im['tag_axis'] is None:
            # 3D data
            idx = im['idx']
            vmin, vmax = im['im'][idx].calculate_clip_range(probs)
        else:
            # 4D data
            idx = im['idx']
            tag = im['tag']
            vmin, vmax = im['im'][tag, idx].calculate_clip_range(probs)
        im['vmin'] = vmin
        im['vmax'] = vmax
        level = (np.float32(vmax) + np.float32(vmin)) / 2
        if np.isnan(level):
            level = 1
        # if abs(level) > 2:
        #     level = round(level)
        im['level'] = level
        window = vmax - vmin
        if np.isnan(window):
            window = 1
        # if abs(window) > 2:
        #     window = round(window)
        im['window'] = window
        im['modified'] = True
        self.update()

    def update_tag(self, value):
        # value = int(round(self.im['slider'].val))
        for key, im in self.im.items():
            if im is not None and im['slider'] is not None:
                inc = 1 / len(im['im'].tags[0])  # Increment per tag step
                tag_index = int(round(int(value / inc) * inc))  # Tag index
                im['tag'] = tag_index
                im['slider'].valtext.set_text(pretty_tag_value(im))
        im['modified'] = True
        self.update()

    def toggle_button(self, button):
        if button == 'Link':
            self.link = self.linkbutton.get_status()[0]  # Link button is button 0
            if self.link:
                # Display same slice for both images
                self.im2['idx'] = self.im['idx']
                self.update()

    def on_press(self, event):
        # Button press - determine action
        if event.button == 1 and not event.dblclick:
            self.start_window_level(event)

    def on_release(self, event):
        # Button release - determine action
        if event.button == 1:
            self.end_window_level(event)

    def on_motion(self, event):
        # Motion - determine action
        if event.button == 1:
            self.modify_window_level(event)

    def start_window_level(self, event):
        # On button press we will see of the mouse is over us and store some data
        im = self.find_image_from_event(event.inaxes)
        if im is not None:
            im['press'] = event.xdata, event.ydata

    def end_window_level(self, event):
        # On button release
        im = self.find_image_from_event(event.inaxes)
        if im is not None:
            im['press'] = None

    def modify_window_level(self, event):
        # On motion, modify window and level, and update display
        im = self.find_image_from_event(event.inaxes)
        if im is not None and im['press'] is not None:
            # delta = (im['vmax'] - im['vmin']) / 100
            delta = (im['im'].max() - im['im'].min()) / 100
            dx = delta * (event.xdata - im['press'][0])
            dy = delta * (im['press'][1] - event.ydata)
            im['press'] = event.xdata, event.ydata
            im['window'] = max(POSITIVE_EPS, im['window'] + dy)
            assert im['window'] >= 0, "Window must be non-negative."
            im['level'] = im['level'] + dx
            im['vmin'] = im['level'] - im['window'] / 2
            im['vmax'] = im['level'] + im['window'] / 2
            im['vmin'], im['vmax'] = _check_vmin_vmax(im['vmin'], im['vmax'], im['norm'])
            im['modified'] = True
            self.update()


class MyPolygonSelector(PolygonSelector):
    """Select a polygon region of an axes.

    Place vertices with each mouse click, and make the selection by completing
    the polygon (clicking on the first vertex). Hold the *ctrl* key and click
    and drag a vertex to reposition it (the *ctrl* key is not necessary if the
    polygon has already been completed). Hold the *shift* key and click and
    drag anywhere in the axes to move all vertices. Press the *esc* key to
    start a new polygon.

    For the selector to remain responsive you must keep a reference to it.

    Class MyPolygonSelector subclasses matplotlib.widgets.PolygonSelector.
    Allows to set an initial polygon.
    """

    def __init__(self, ax, onselect, useblit=False,
                 lineprops=None, markerprops=None, vertex_select_radius=10,
                 vertices=None, tag=None, copy=None, paste=None):
        super().__init__(ax, onselect, useblit=useblit,
                         props=lineprops,
                         handle_props=markerprops,
                         grab_range=vertex_select_radius)

        self.tag = tag
        self.copy_handle = copy
        self.paste_handle = paste
        self._polygon_completed = False

        if lineprops is None:
            self.lineprops = dict(color='k', linestyle='-', linewidth=2, alpha=0.5)
        else:
            self.lineprops = lineprops
        self.lineprops['animated'] = self.useblit

        if markerprops is None:
            self.markerprops = dict(markeredgecolor='k',
                                    markerfacecolor=self.lineprops.get('color', 'k'))
        else:
            self.markerprops = markerprops

        self.vertex_select_radius = vertex_select_radius

        if vertices is not None and len(vertices):
            self.verts = vertices

    def _on_key_release(self, event):
        """Key release event handler."""
        # Add back the pending vertex if leaving the 'move_vertex' or
        # 'move_all' mode (by checking the released key)
        if (not self._selection_completed
                and
                (event.key == self._state_modifier_keys.get('move_vertex')
                 or event.key == self._state_modifier_keys.get('move_all'))):
            self._xys.append((event.xdata, event.ydata))
            self._draw_polygon()
        # Reset the polygon if the released key is the 'clear' key.
        elif event.key == self._state_modifier_keys.get('clear'):
            event = self._clean_event(event)
            self._xys = [(event.xdata, event.ydata)]
            self._selection_completed = False
            self._remove_box()
            self.set_visible(True)
        # Copy polygon to paste buffer using handle
        elif event.key.upper() == 'C':
            if self.copy_handle is not None:
                self.copy_handle(self)
        # Add polygon from paste buffer handle
        elif event.key.upper() == 'V':
            if self.paste_handle is not None:
                obj = self.paste_handle()
                self.verts = obj.verts
                self._selection_completed = obj._selection_completed
                self._draw_polygon()
                self.set_visible(True)
                if self._selection_completed:
                    self.onselect(self.verts)


def default_layout(fig, n):
    """Setup a default layout for given number of axes.

    Args:
        fig: matplotlib figure
        n: Number of axes required
    Returns:
        List of Axes
    Raises:
        ValueError: When no ax axes or > 9*9 are required. When no figure is given.
    """
    if fig is None:
        raise ValueError("No Figure given")
    if n < 1:
        raise ValueError("No layout when no axes are required")
    for rows in range(1, 5):
        if rows * rows >= n:
            return fig.subplots(rows, rows, squeeze=False)  # columns = rows
        if rows * (rows + 1) >= n:
            return fig.subplots(rows, rows + 1, squeeze=False)  # columns = rows+1
    raise ValueError("Too many axes required (n={})".format(n))


def grid_from_roi(im: Series, vertices: dict, single: bool = False) -> Union[bool, Series]:
    """Return drawn ROI as grid.

    Args:
        im (imagedata.Series): Series object as template
        vertices: The polygon vertices, as a dictionary of tags of (x,y)
        single (bool): Draw ROI in single slice per tag
    Returns:
        Series with shape (nz,ny,nx) from original image, dtype ubyte.
        Voxels inside ROI is 1, 0 outside.
    """

    def _roi_in_any_slice(tag):
        """Check whether there is a ROI in any slice"""
        t, i = tag
        for idx in range(im.slices):
            # if (t, idx) in vertices and vertices[t, idx] is None:
            #     print('Check {} None'.format((t, idx)))
            # elif (t, idx) in vertices:
            #     print('Check {} {}'.format((t, idx), len(vertices[t, idx])))
            # else:
            #     print('Check {} not found'.format((t, idx)))
            if (t, idx) in vertices and vertices[t, idx] is not None:
                return True
        return False

    keys = list(vertices.keys())[0]
    # print('Viewer.grid_from_roi: keys: {}'.format(keys))
    # print('Viewer.grid_from_roi: vertices: {}'.format(vertices))
    follow = issubclass(type(keys), tuple)
    nt, nz, ny, nx = len(im.tags[0]), im.slices, im.rows, im.columns
    input_order = im.input_order
    if follow and not single:
        grid = np.zeros_like(im, dtype=np.ubyte)
        skipped = []
        copied = []
        for idx in range(nz):
            last_used_tag = None
            for t in range(nt):
                tag = t, idx
                if tag not in vertices or vertices[tag] is None:
                    if last_used_tag is None:
                        # Most probably a slice with no ROIs
                        skipped.append(tag)
                        continue
                    elif _roi_in_any_slice(tag):
                        # print('Found in some slice for', tag)
                        skipped.append(tag)
                        continue
                    # Propagate last drawn ROI to unfilled tags
                    vertices[tag] = copy.copy(vertices[last_used_tag])
                    copied.append((tag, last_used_tag))
                else:
                    last_used_tag = tag
                path = MplPath(vertices[tag])
                x, y = np.meshgrid(np.arange(nx), np.arange(ny))
                x, y = x.flatten(), y.flatten()
                points = np.vstack((x, y)).T
                grid[t, idx] = path.contains_points(points).reshape((ny, nx))
    elif follow and single:
        grid = np.zeros_like(im, dtype=np.ubyte)
        skipped = []
        copied = []
        last_used_tag = None
        for t in range(nt):
            for idx in range(nz):
                tag = t, idx
                if tag not in vertices or vertices[tag] is None:
                    if last_used_tag is None:
                        # Most probably a slice with no ROIs
                        skipped.append(tag)
                        continue
                    elif last_used_tag[1] != idx:
                        continue
                    elif _roi_in_any_slice(tag):
                        # print('Found in some slice for', tag)
                        skipped.append(tag)
                        continue
                    # Propagate last drawn ROI to unfilled tags
                    vertices[tag] = copy.copy(vertices[last_used_tag])
                    copied.append((tag, last_used_tag))
                else:
                    last_used_tag = tag
                path = MplPath(vertices[tag])
                x, y = np.meshgrid(np.arange(nx), np.arange(ny))
                x, y = x.flatten(), y.flatten()
                points = np.vstack((x, y)).T
                grid[t, idx] = path.contains_points(points).reshape((ny, nx))
        # if len(skipped) > 0:
        #     print('Skipped: {}'.format(skipped))
        # if len(copied) > 0:
        #     print('Copied: {}'.format(copied))
    else:
        grid = np.zeros((nz, ny, nx), dtype=np.ubyte)
        for idx in range(nz):
            if idx not in vertices or vertices[idx] is None:
                continue
            path = MplPath(vertices[idx])
            x, y = np.meshgrid(np.arange(nx), np.arange(ny))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x, y)).T
            grid[idx] = path.contains_points(points).reshape((ny, nx))
        input_order = 'none'
    if im.ndim == 2:
        grid = grid.reshape((ny, nx))
    return Series(grid, input_order=input_order, template=im, geometry=im)


def get_level(si, level):
    if level is None:
        # First, attempt to get DICOM attribute
        try:
            level = si.windowCenter
        except (KeyError, AttributeError, TypeError):
            pass
        try:
            if len(level) > 1:
                level = level[0]
        except TypeError:
            pass
    if level is None:
        level = (np.float32(np.nanmax(si)) + np.float32(np.nanmin(si))) / 2
        if np.isnan(level):
            level = 1
        if abs(level) > 2:
            level = round(level)
    return level


def get_window_level(si, norm, window, level):
    if window is None:
        # First, attempt to get DICOM attribute
        try:
            window = si.windowWidth
        except (KeyError, AttributeError, TypeError):
            pass
        try:
            if len(window) > 1:
                window = window[0]
            # print('Viewer.get_window_level: {} len {}'.format(window, len(window)))
        except TypeError:
            pass
    if window is None:
        window = np.float32(np.nanmax(si)) - np.float32(np.nanmin(si))
        if np.isnan(window):
            window = 1
        if abs(window) > 2:
            window = round(window)
    level = get_level(si, level)
    vmin, vmax = _check_vmin_vmax(level - window / 2, level + window / 2, norm)
    return window, level, vmin, vmax


def _check_vmin_vmax(vmin, vmax, norm):
    if type(norm) is type:
        norm = norm(vmin=vmin, vmax=vmax)
    if type(norm) is matplotlib.colors.LogNorm:
        vmin = max(POSITIVE_EPS, vmin)
        vmax = max(POSITIVE_EPS, vmin, vmax)
    return vmin, vmax


def build_info(im, colormap, norm, colorbar, window, level):
    _name: str = '{}.{}'.format(__name__, build_info.__name__)

    if im is None:
        return None
    if not issubclass(type(im), Series):
        raise ValueError('Cannot display image of type {}'.format(type(im)))

    # im might be modified below (color version), hence, save present color presentation
    im_color = im.color
    im_colormap = copy.copy(im.colormap)
    im_colormap_norm = copy.copy(im.colormap_norm)
    try:
        im_colormap_label = im.colormap_label
    except ValueError:
        im_colormap_label = None

    if colormap is None:
        colormap = 'Greys_r'
    try:
        if colormap == 'Greys_r' and im.photometricInterpretation == 'MONOCHROME1':
            colormap = 'Greys'
    except ValueError:
        pass
    if np.issubdtype(im.dtype, np.floating):
        lut = 256
    elif np.issubdtype(im.dtype, np.complexfloating):
        lut = 256
        logger.warning('{}: Displaying real part of complex values.'.format(_name))
        im = np.real(im)
    elif im.color:
        lut = 256
        im = im.view(dtype=np.uint8).reshape(im.shape + (3,))
    else:
        lut = (np.nanmax(im).item()) + 1
    if im_colormap is None:
        if not issubclass(type(colormap), matplotlib.colors.Colormap):
            colormap = plt.get_cmap(colormap, lut)
        colormap.set_bad(color='k')  # Important for log display of non-positive values
        colormap.set_under(color='k')
        colormap.set_over(color='w')
    else:
        colormap = im_colormap
    window, level, vmin, vmax = get_window_level(im, norm, window, level)
    if im_colormap_norm is None:
        if type(norm) is type:
            norm = norm(vmin=vmin, vmax=vmax)
        if im_color:
            norm = None
    else:
        norm = im_colormap_norm
    tag_axis = im.get_tag_axis()
    slice_axis = im.get_slice_axis()

    try:
        tags = len(im.tags[0])
    except TypeError:
        tags = 0

    return {
        'im': im,  # Image Series instance
        'input_order': im.input_order,
        'color': im_color,
        'modified': True,  # update()
        'show_text': True,  # Show text on display
        'artists': [],  # List of artists
        'slider': None,  # 4D slider
        'lower_left_text': None,  # AnchoredText object
        'lower_left_data': None,  # Tuple of present data
        'lower_right_text': None,  # AnchoredText object
        'lower_right_data': None,  # Tuple of present data
        'scrollable': im.slices > 1,  # Can we scroll the instance?
        'taggable': tag_axis is not None,  # Can we slide through tags?
        'tags': tags,  # Number of tags
        'slices': im.slices,  # Number of slices
        'rows': im.rows,  # Number of rows
        'columns': im.columns,  # Number of columns
        'tag': 0,  # Displayed tag index
        'idx': im.slices // 2,  # Displayed slice index
        'tag_axis': tag_axis,  # Axis instance of im
        'slice_axis': slice_axis,  # Axis instance of im
        'colormap': colormap,  # Colour map
        'colormap_label': im_colormap_label,  # Colour map label
        'norm': im_colormap_norm if im_color else norm,  # Normalization function
        'colorbar': im_colormap,  # Display colorbar unless RGB image
        'window': window,  # Window center
        'level': level,  # Window level
        'vmin': vmin,  # Lower window value
        'vmax': vmax  # Upper window value
    }


def pretty_tag_value(im):
    tag = im['tag']
    if im['input_order'] == 'time':
        return '{0:0.2f}s'.format(im['im'].timeline[tag])
    elif im['input_order'] == 'b':
        return '{}'.format(int(im['im'].tags[0][tag]))
    elif im['input_order'] == 'te':
        return '{}ms'.format(int(im['im'].tags[0][tag]))
    elif im['input_order'] == 'fa':
        return '{}'.format(im['im'].tags[0][tag])
    else:
        return '{}'.format(im['im'].tags[0][tag])


def pretty_window_level(im):
    si, window, level = im['im'], im['window'], im['level']
    if si.dtype.kind in ('i', 'u'):
        fmt = 'SL: {0:d}\nW: {1:d} C: {2:d}'
        window = int(window)
        level = int(level)
    else:
        fmt = 'SL: {0:d}\nW: {1:.2f} C: {2:.2f}'
        window = np.around(window, 2)
        level = np.around(level, 2)
    return fmt, window, level
