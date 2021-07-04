import copy
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.widgets import PolygonSelector, ToolHandles
from matplotlib.lines import Line2D
# from matplotlib.widgets import Slider
from matplotlib.path import Path as MplPath
import numpy as np

from imagedata.series import Series


def get_slice_axis(im):
    try:
        return im.find_axis('slice')
    except ValueError:
        return None


def get_tag_axis(im):
    try:
        return im.find_axis(im.input_order)
    except ValueError:
        return None


def get_level(si, level):
    if level is None:
        # First, attempt to get DICOM attribute
        try:
            level = si.getDicomAttribute('WindowCenter')
        except (KeyError, AttributeError):
            pass
    if level is None:
        level = (si.max() - si.min()) / 2
    return level


def get_window_level(si, window, level):
    if window is None:
        # First, attempt to get DICOM attribute
        try:
            window = si.getDicomAttribute('WindowWidth')
        except (KeyError, AttributeError):
            pass
    if window is None:
        window = si.max() - si.min()
    level = get_level(si, level)
    vmin = level - window / 2
    vmax = level + window / 2
    return window, level, vmin, vmax


def build_info(im, cmap, window, level):
    if im is None:
        return None
    if not issubclass(type(im), Series):
        raise ValueError('Cannot display image of type {}'.format(type(im)))
    if cmap is None:
        cmap = 'gray'
    window, level, vmin, vmax = get_window_level(im, window, level)
    tag_axis = get_tag_axis(im)
    slice_axis = get_slice_axis(im)

    return {
        'im': im,  # Image Series instance
        'input_order': im.input_order,
        'modified': True,  # update()
        'slider': None,  # 4D slider
        'lower_left_text': None,  # AnchoredText object
        'lower_left_data': None,  # Tuple of present data
        'lower_right_text': None,  # AnchoredText object
        'lower_right_data': None,  # Tuple of present data
        'scrollable': im.slices > 1,  # Can we scroll the instance?
        'taggable': tag_axis is not None,  # Can we slide through tags
        'tags': len(im.tags[0]),  # Number of tags
        'slices': im.slices,  # Number of slices
        'rows': im.rows,  # Number of rows
        'columns': im.columns,  # Number of columns
        'tag': 0,  # Displayed tag index
        'idx': im.slices // 2,  # Displayed slice index
        'tag_axis': tag_axis,  # Axis instance of im
        'slice_axis': slice_axis,  # Axis instance of im
        'cmap': cmap,  # Colour map
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


class Viewer:
    """Viewer -- a graphical tool to display and interact with Series objects.

    Args:
        images (Series or list): Series object or list of Series objects to view.
        fig (Figure): matplotlib.plt.figure if already exist (optional).
        ax (Axes): matplotlib axis if already exist (optional).
        follow (bool): Copy ROI to next tag. Default: False.
        cmap (str): Colour map for display. Default: gray.
        windows (number): Window width of signal intensities. Default: DICOM Window Width.
        level (number): Window level of signal intensities. Default: DICOM Window Center.
        link (bool): Whether scrolling is linked between displayed objects. Default: False.

    """

    def __init__(self, images, fig=None, ax=None, follow=False,
                 cmap='gray', window=None, level=None, link=False):
        self.fig = fig
        self.ax = ax
        if self.ax is None:
            self.ax = default_layout(fig, len(images))
        self.im = {}
        for i, im in enumerate(images):
            self.im[i] = build_info(im, cmap, window, level)
        self.follow = follow
        self.link = link
        self.cidenter = None
        self.cidleave = None
        self.cidscroll = None
        self.vertices = None  # The polygon vertices, as a dictionary of tags of (x,y)
        self.poly = None
        self.viewport = {}
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
        vp_idx = 0
        for row in range(rows):
            for column in range(columns):
                if vp_idx in self.im:
                    self.viewport[vp_idx] = {
                        'ax': axes[row, column],
                        'present': None,
                        'next': vp_idx,
                        'h': None}
                else:
                    self.viewport[vp_idx] = None
                    axes[row, column].set_axis_off()
                vp_idx += 1

    def update(self):
        # For each viewport
        for vp_idx in self.viewport:
            vp = self.viewport[vp_idx]
            if vp is None:
                continue
            if vp['next'] != vp['present']:
                # We want to show another image in this viewport
                if vp['next'] in self.im:
                    vp['h'] = self.show(vp['ax'], self.im[vp['next']])
                    vp['present'] = vp['next']
                else:
                    raise IndexError("Series {} should be viewed, but does not exist".format(
                        vp['next']
                    ))
            # Update present image in viewport
            im = self.im[vp['present']]
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
            fmt = 'SL: {0:d}\nW: {1:d} C: {2:d}'
            window = int(im['window'])
            level = int(im['level'])
            if im['lower_left_text'] is not None and im['lower_left_data'] != (window, level, im['idx']):
                im['lower_left_text'].txt.set_text(fmt.format(im['idx'], window, level))
                im['lower_left_data'] = (window, level, im['idx'])
            vp['ax'].axes.figure.canvas.draw()
            im['modified'] = False

    def show(self, ax, im):
        if im is None:
            return None
        if im['slice_axis'] is None:
            # 2D viewer
            h = ax.imshow(im['im'], cmap=im['cmap'], vmin=im['vmin'], vmax=im['vmax'])
        elif im['tag_axis'] is None:
            # 3D viewer
            h = ax.imshow(im['im'][im['idx'], ...], cmap=im['cmap'], vmin=im['vmin'], vmax=im['vmax'])
        else:
            # 4D viewer
            h = ax.imshow(im['im'][im['tag'], im['idx'], ...], cmap=im['cmap'], vmin=im['vmin'], vmax=im['vmax'])
            # EA# h.figure.subplots_adjust(bottom=0.15)
            # EA# ax_tag = plt.axes([0.23, 0.02, 0.56, 0.04])
            # EA# im['slider'] = Slider(ax_tag, im['input_order'], 0, im['im'].shape[0] - 1, valinit=im['tag'], valstep=1)
            # EA# im['slider'].on_changed(self.update_tag)
            # Lower right text
            fmt = '{}[{}]: {}'.format(im['input_order'], im['tag'], pretty_tag_value(im))
            im['lower_right_data'] = (im['tag'],)
            im['lower_right_text'] = AnchoredText(fmt,
                                                  prop=dict(size=6, color='white', backgroundcolor='black'),
                                                  frameon=False,
                                                  loc='lower right'
                                                  )
            ax.add_artist(im['lower_right_text'])
        fmt = 'SL: {0:d}\nW: {1:d} C: {2:d}'
        window = int(im['window'])
        level = int(im['level'])
        im['lower_left_data'] = (window, level, im['idx'])
        im['lower_left_text'] = AnchoredText(fmt.format(im['idx'], window, level),
                                             prop=dict(size=6, color='white', backgroundcolor='black'),
                                             frameon=False,
                                             loc='lower left'
                                             )
        ax.add_artist(im['lower_left_text'])
        im['modified'] = True

        ax.set_axis_off()
        # if im['slices'] == im2['slices']:
        #    plt.subplots_adjust(bottom=0.1)
        #    self.rax = plt.axes([0.0, 0.0, 0.2, 0.1], frame_on=False)
        #    self.linkbutton = CheckButtons(self.rax, ['Link'], [link])
        #    self.linkclicked = self.linkbutton.on_clicked(self.toggle_button)
        return h

    def connect_draw(self, roi=None, color='w'):
        self.poly_color = color
        idx = self.im[0]['idx']
        if roi is None:
            self.poly = {}
            self.vertices = {}
            if self.follow:
                self.poly[0, idx] = MyPolygonSelector(self.ax[0,0], self.onselect, lineprops={'color': self.poly_color})
            else:
                self.poly[idx] = MyPolygonSelector(self.ax[0,0], self.onselect, lineprops={'color': self.poly_color})
        else:
            self.poly = {}
            self.vertices = roi
            if self.follow:
                for tag in range(self.im[0]['tags']):
                    for i in range(self.im[0]['slices']):
                        verts = None
                        if (tag, i) in self.vertices:
                            verts = self.vertices[tag, i]
                        self.poly[tag, i] = MyPolygonSelector(self.ax[0,0], self.onselect, lineprops={'color': self.poly_color}, verts=verts)
                        # Polygon on single slice and tag 0, only
                        if i == idx and tag == 0:
                            self.poly[tag, i].connect_default_events()
                            self.poly[tag, i].set_visible(True)
                            self.poly[tag, i].update()
                        else:
                            self.poly[tag, i].disconnect_events()
                            self.poly[tag, i].set_visible(False)
                            self.poly[tag, i].update()
            else:
                for i in range(self.im[0]['slices']):
                    verts = None
                    if i in self.vertices:
                        verts = self.vertices[i]
                    self.poly[i] = MyPolygonSelector(self.ax[0,0], self.onselect, lineprops={'color': self.poly_color}, verts=verts)
                    # Polygon on single slice only
                    if i != idx:
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

    def onselect(self, verts):
        idx = self.im[0]['idx']
        if self.follow:
            tag = self.im[0]['tag']
            self.vertices[tag, idx] = copy.copy(verts)
        else:
            self.vertices[idx] = copy.copy(verts)

    # def grid_from_roi(self):
    #     """Return drawn ROI as grid.
    #
    #     Returns:
    #         Numpy ndarray with shape (nz,ny,nx) from original image, dtype ubyte.
    #         Voxels inside ROI is 1, 0 outside.
    #     """
    #     nt, nz, ny, nx = self.im[0]['tags'], self.im[0]['slices'], self.im[0]['rows'], self.im[0]['columns']
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
        return self.vertices

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
            self.viewport_advance(event.inaxes, 1)
        elif event.key == 'pagedown':
            self.viewport_advance(event.inaxes, -1)
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
        im['modified'] = old_idx != im['idx']
        if self.poly is not None:
            new_idx = im['idx']
            if self.follow:
                old_idx = im['tag'], old_idx
                new_idx = im['tag'], new_idx
            if im['modified']:
                self.poly[old_idx].disconnect_events()
                self.poly[old_idx].set_visible(False)
                self.poly[old_idx].update()
            if new_idx in self.poly and self.poly[new_idx] is not None:
                # self.poly[im['idx']].connect_event(event, self.onselect)
                self.poly[new_idx].connect_default_events()
                self.poly[new_idx].set_visible(True)
                self.poly[new_idx].update()
            else:
                self.poly[new_idx] = MyPolygonSelector(self.ax[0,0], self.onselect, lineprops={'color': self.poly_color})
        # if self.link and self.im['scrollable'] and self.im2['scrollable']:
        #    self.im['idx'] = min(max(self.im['idx'] + increment, 0), self.im['slices']-1)
        #    self.im2['idx'] = self.im['idx']
        # elif inaxes == self.im['ax'] and self.im['scrollable']:
        #    self.im['idx'] = min(max(self.im['idx'] + increment, 0), self.im['slices']-1)
        # elif self.im2 is not None and inaxes == self.im2['ax'] and self.im2['scrollable']:
        #    self.im2['idx'] = min(max(self.im2['idx'] + increment, 0), self.im2['slices']-1)
        self.update()

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
            self.onselect(self.poly[old_tag, idx].verts)
            if (new_tag, idx) not in self.poly and (old_tag, idx) in self.poly and self.poly[old_tag, idx] is not None:
                # Copy the polygon to next tag when there is none
                self.poly[new_tag, idx] = MyPolygonSelector(self.ax[0,0], self.onselect,
                                                            lineprops={'color': self.poly_color},
                                                            verts=self.poly[old_tag, idx].verts)
            self.poly[old_tag, idx].disconnect_events()
            self.poly[old_tag, idx].set_visible(False)
            self.poly[old_tag, idx].update()
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

    def viewport_advance(self, inaxes, increment):
        viewports = len(self.viewport.keys())
        # for vp_idx in range(viewports):
        #    if vp_idx in self.viewport:
        #        print('enter', self.viewport[vp_idx]['next'])
        images = len(self.im)
        # print('viewport_advance: viewports {}'.format(viewports))
        vp_idx = 0
        if increment == 1:
            if self.viewport[viewports - 1] is None:
                return
            next_im = self.viewport[viewports - 1]['present'] + increment
            if next_im >= images:
                return  # Don't move outside range of series
            # Drop first series, move other series forward
            for vp_idx in range(viewports - 1):
                # print('increment vp_idx: {}'.format(vp_idx))
                self.viewport[vp_idx]['present'] = None
                self.viewport[vp_idx]['next'] = self.viewport[vp_idx + 1]['present']
                self.viewport[vp_idx]['h'] = None
            # Append new series when available
            self.viewport[viewports - 1]['next'] = next_im
            self.viewport[viewports - 1]['present'] = None
            self.viewport[viewports - 1]['h'] = None
        elif increment == -1:
            next_im = self.viewport[0]['present'] + increment
            if next_im < 0:
                return  # Don't move in-front of first image
            # Move other series backwards
            for vp_idx in range(viewports - 1, 0, -1):
                # print('decrement vp_idx: {}'.format(vp_idx))
                self.viewport[vp_idx]['next'] = self.viewport[vp_idx - 1]['present']
                self.viewport[vp_idx]['present'] = None
                self.viewport[vp_idx]['h'] = None
            # Insert new series at front when available
            self.viewport[0]['next'] = next_im
            self.viewport[0]['present'] = None
            self.viewport[0]['h'] = None
        else:
            raise ValueError('Increment shall be +/-1')
        # im['modified'] = True
        # for vp_idx in range(viewports):
        #    if vp_idx in self.viewport:
        #        print('leave', self.viewport[vp_idx]['next'])
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
            dx = 10 * (event.xdata - im['press'][0])
            dy = 10 * (im['press'][1] - event.ydata)
            im['press'] = event.xdata, event.ydata
            im['window'] = max(1e-3, im['window'] + dx)
            im['level'] = im['level'] + dy
            im['vmin'] = im['level'] - im['window'] / 2
            im['vmax'] = im['level'] + im['window'] / 2
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
                 lineprops=None, markerprops=None, vertex_select_radius=15, verts=None):
        super().__init__(ax, onselect, useblit=useblit,
                         lineprops=lineprops, markerprops=markerprops, vertex_select_radius=vertex_select_radius)

        if verts is not None and len(verts):
            self._xs = [x for x,y in verts]
            self._ys = [y for x,y in verts]
            # Append end-point of closed polygon
            self._xs.append(self._xs[0])
            self._ys.append(self._ys[0])
            self._polygon_completed = True

            if lineprops is None:
                lineprops = dict(color='k', linestyle='-', linewidth=2, alpha=0.5)
            lineprops['animated'] = self.useblit
            self.line = Line2D(self._xs, self._ys, **lineprops)
            self.ax.add_line(self.line)

            if markerprops is None:
                markerprops = dict(markeredgecolor='k',
                                   markerfacecolor=lineprops.get('color', 'k'))
            self._polygon_handles = ToolHandles(self.ax, self._xs, self._ys,
                                                useblit=self.useblit,
                                                marker_props=markerprops)

            self._active_handle_idx = -1
            self.vertex_select_radius = vertex_select_radius

            self.artists = [self.line, self._polygon_handles.artist]
            self.set_visible(True)


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


def grid_from_roi(im, vertices):
    """Return drawn ROI as grid.

    Args:
        im (Series): Series object as template
        vertices: The polygon vertices, as a dictionary of tags of (x,y)
    Returns:
        Numpy ndarray with shape (nz,ny,nx) from original image, dtype ubyte.
        Voxels inside ROI is 1, 0 outside.
    """
    keys = list(vertices.keys())[0]
    # print('grid_from_roi: keys: {}'.format(keys))
    # print('grid_from_roi: vertices: {}'.format(vertices))
    follow = issubclass(type(keys), tuple)
    nt, nz, ny, nx = im.shape
    if follow:
        grid = np.zeros_like(im, dtype=np.ubyte)
        for idx in range(nz):
            last_used_tag = None
            for t in range(nt):
                tag = t, idx
                if tag not in vertices or vertices[tag] is None:
                    if last_used_tag is None:
                        # Most probably a slice with no ROIs
                        continue
                    # Propagate last drawn ROI to unfilled tags
                    vertices[tag] = vertices[last_used_tag]
                else:
                    last_used_tag = tag
                path = MplPath(vertices[tag])
                x, y = np.meshgrid(np.arange(nx), np.arange(ny))
                x, y = x.flatten(), y.flatten()
                points = np.vstack((x, y)).T
                grid[t, idx] = path.contains_points(points).reshape((ny, nx))
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
    return Series(grid, input_order=im.input_order, template=im, geometry=im)

