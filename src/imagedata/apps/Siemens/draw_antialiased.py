"""Library to draw an antialiased line.
  http://stackoverflow.com/questions/3122049/drawing-an-anti-aliased-line-with-thepython-imaging-library
  https://en.wikipedia.org/wiki/Xiaolin_Wu%27s_line_algorithm

  https://yellowsplash.wordpress.com/2009/10/23/fast-antialiased-circles-and-ellipses-from-xiaolin-wus-concepts/
  https://stackoverflow.com/questions/37589165/drawing-an-antialiased-circle-as-described-by-xaolin-wu#37714284
"""

import math
import numpy as np
from matplotlib.path import Path


def draw_polygon_mask(canvas, points, colour, threshold, fill=True):
    """Make a 2D mask [ny,nx] on canvas

    Optionally, fill the polygon with the same colour.

    Args:
        canvas: numpy array
        points: polygon points (list of tuples (x,y)) (float)
        colour: colour to draw (mask index) (int)
        threshold: only pixels with an alpha > threshold will be drawn (float, 0.0-1.0)
        fill: whether to fill the circle (boolean)
    """
    # points_matrix = self.get_points_matrix(hdr)
    """
    print("polygon: points cm    ", points)
    print("polygon: points matrix", polygon)
    print("polygon: colour %d fill" % colour, fill)
    """

    # Flag the voxels that will be in the mask
    mask = np.zeros_like(canvas, dtype=np.bool)

    # Colour the voxels on the polygon with the True colour
    pn = points[len(points) - 1]  # Close the polygon
    for p in points:
        x1, y1 = pn
        x2, y2 = p
        draw_line_mask(mask, x1, y1, x2, y2, True, threshold)
        pn = p
    # canvas.save("/tmp/polygon.png", "PNG")

    if fill:
        # Colour the voxels inside the polygon with the True colour
        inside = point_in_polygon(mask, points)
        mask = np.logical_or(mask, inside)
    # Set voxels in the mask to the given colour
    canvas[mask] = colour


def plot(canvas, x, y, steep, colour, alpha, threshold):
    """Draws an anti-aliased pixel on a line."""
    if steep:
        x, y = y, x
    # if x < canvas.shape[1] and y < canvas.shape[0] and x >= 0 and y >= 0:
    if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
        x = int(x)
        y = int(y)
        if alpha >= threshold:
            canvas[y, x] = colour


def iround(x):
    """Rounds x to the nearest integer."""
    return ipart(x + 0.5)


def ipart(x):
    """Floors x."""
    return math.floor(x)


def fpart(x):
    """Returns the fractional part of x."""
    return x - math.floor(x)


def rfpart(x):
    """Returns the 1 minus the fractional part of x."""
    return 1 - fpart(x)


def draw_line_mask(canvas, x1, y1, x2, y2, colour, threshold):
    """Draw line mask on NumPy array.

    Apply the Xialon Wu anti-aliasing algorithm for drawing line.
    Draw points only when the alpha blending is above a set threshold.
    Only given colour value is drawn. Intended usage is as a mask index.

    Args:
        canvas: numpy array (2D)
        (x1,y1), (x2,y2): line end points (float)
        colour: colour to draw (mask index) (int)
        threshold: only pixels with an alpha > threshold will be drawn (float, 0.0-1.0)
    """
    dx = x2 - x1
    """
    if not dx:
        # Vertical line
        draw_line((x1, y1, x2, y2), fill=col, width=1)
        return
    """

    dy = y2 - y1
    steep = abs(dx) < abs(dy)
    if steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
        dx, dy = dy, dx
    if x2 < x1:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    try:
        gradient = float(dy) / float(dx)
    except ZeroDivisionError:
        gradient = 1.0

    # Handle first endpoint
    xend = round(x1)
    yend = y1 + gradient * (xend - x1)
    xgap = rfpart(x1 + 0.5)
    xpxl1 = xend  # this will be used in the main loop
    ypxl1 = ipart(yend)
    plot(canvas, xpxl1, ypxl1, steep, colour, rfpart(yend) * xgap, threshold)
    plot(canvas, xpxl1, ypxl1 + 1, steep, colour, fpart(yend) * xgap, threshold)
    intery = yend + gradient  # first y-intersection for the main loop

    # handle second endpoint
    xend = round(x2)
    yend = y2 + gradient * (xend - x2)
    xgap = fpart(x2 + 0.5)
    xpxl2 = xend  # this will be used in the main loop
    ypxl2 = ipart(yend)
    plot(canvas, xpxl2, ypxl2, steep, colour, rfpart(yend) * xgap, threshold)
    plot(canvas, xpxl2, ypxl2 + 1, steep, colour, fpart(yend) * xgap, threshold)

    # main loop
    for x in range(int(xpxl1 + 1), int(xpxl2)):
        plot(canvas, x, ipart(intery), steep, colour, rfpart(intery), threshold)
        plot(canvas, x, ipart(intery) + 1, steep, colour, fpart(intery), threshold)
        intery = intery + gradient


"""
def draw_simple_line(x1, y1, x2, y2, fill=col, width=1):
    if x1 != x2:
        # Horizontal line
        assert y1 == y2, "Horizontal line only."
        if x1 > x2: x2,x1 = x1,x2
        x = x1
        while x <= x2:
            plot(canvas, x, y1
            x += 1
"""


def point_in_polygon(canvas, polygon):
    ny, nx = canvas.shape

    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x, y)).T

    path = Path(polygon)
    grid = path.contains_points(points)
    grid = grid.reshape((ny, nx))

    return grid


def draw_circle_mask(canvas, center_x, center_y, outer_radius, colour, threshold, fill=True):
    """Draw circle mask on NumPy array.

    Apply algorithm for drawing anti-aliased circle.
    Draw points only when the alpha blending is above a set threshold.
    Only given _colour value is drawn. Intended usage is as a mask index.

    Optionally, fill the circle with the same _colour.

    Reference:
      https://stackoverflow.com/questions/37589165/drawing-an-antialiased-circle-as-described-by-xaolin-wu#37714284

    Args:
        canvas: numpy array
        center_x, center_y: center of circle in array coordinates (int)
        outer_radius: radius of circle in array dimension (float)
        colour: _colour to draw (mask index) (int)
        threshold: only pixels with an alpha > threshold will be drawn (float, 0.0-1.0)
        fill: whether to fill the circle (boolean)
    """
    """
    def _draw_8point(_canvas, _cx, _cy, x, y, _colour):
        # Draw the 8 symmetries
        print("_draw_8point", _cy, _cx, y, x)
        print("_draw_8point", _cy + y,  _cx - x)
        print("_draw_8point", _cy + y,  _cx + x)
        print("_draw_8point", _cy - y,  _cx - x)
        print("_draw_8point", _cy - y,  _cx + x)
        print("_draw_8point", _cx + x,  _cy - y)
        print("_draw_8point", _cx + x,  _cy + y)
        print("_draw_8point", _cx - x,  _cy - y)
        print("_draw_8point", _cx - x,  _cy + y)

        _canvas[_cy + y,  _cx - x] = _colour
        _canvas[_cy + y,  _cx + x] = _colour
        _canvas[_cy - y,  _cx - x] = _colour
        _canvas[_cy - y,  _cx + x] = _colour
        _canvas[_cx + x,  _cy - y] = _colour
        _canvas[_cx + x,  _cy + y] = _colour
        _canvas[_cx - x,  _cy - y] = _colour
        _canvas[_cx - x,  _cy + y] = _colour
    """

    def _draw_8point(_canvas, _cx, _cy, _i, _j, _colour):
        """Draws 8 points, one on each octant."""
        # Square symmetry
        local_coord = [(_i * (-1) ** (k % 2), _j * (-1) ** (k // 2)) for k in range(4)]
        # Diagonal symmetry
        local_coord += [(j_, i_) for i_, j_ in local_coord]
        for i_, j_ in local_coord:
            # print("_draw_8point", _cy + j_,  _cx + i_)
            _canvas[_cy + j_, _cx + i_] = _colour

    i = 0
    j = outer_radius
    last_fade_amount = 0
    # fade_amount = 0

    max_opaque = 1.0

    while i < j:
        height = math.sqrt(max(outer_radius * outer_radius - i * i, 0))
        fade_amount = max_opaque * (math.ceil(height) - height)

        if fade_amount < last_fade_amount:
            # Opaqueness reset so drop down a row.
            j -= 1
        last_fade_amount = fade_amount

        # We're fading out the current _j row, and fading in the next one down.
        if max_opaque - fade_amount > threshold:
            _draw_8point(canvas, center_x, center_y, i, j, colour)
        if fade_amount > threshold:
            _draw_8point(canvas, center_x, center_y, i, j - 1, colour)

        i += 1

    if fill:
        boundary_fill4(canvas, center_x, center_y, colour, colour)


def draw_ellipse_mask(canvas, center_x, center_y, outer_radius, colour, threshold, fill=True):
    """Draw ellipse mask on NumPy array.

    Apply algorithm for drawing anti-aliased ellipse.
    Draw points only when the alpha blending is above a set threshold.
    Only given _colour value is drawn. Intended usage is as a mask index.

    Optionally, fill the ellipse with the same _colour.

    Reference:
      https://yellowsplash.wordpress.com/2009/10/23/fast-antialiased-circles-and-ellipses-from-xiaolin-wus-concepts/
      https://stackoverflow.com/questions/37589165/drawing-an-antialiased-circle-as-described-by-xaolin-wu#37714284

    Args:
        canvas: numpy array
        center_x, center_y: center of ellipse in array coordinates (int)
        outer_radius: radius of circle in array dimension (float)
        colour: colour to draw (mask index) (int)
        threshold: only pixels with an alpha > threshold will be drawn (float, 0.0-1.0)
        fill: whether to fill the circle (boolean)
    """

    def _draw_4point(_canvas, _cx, _cy, x, y, _colour):
        # Draw the 8 symmetries
        print("_draw_8point", _cy, _cx, y, x)
        print("_draw_8point", _cy + y, _cx - x)
        print("_draw_8point", _cy + y, _cx + x)
        print("_draw_8point", _cy - y, _cx - x)
        print("_draw_8point", _cy - y, _cx + x)

        _canvas[_cy + y, _cx - x] = _colour
        _canvas[_cy + y, _cx + x] = _colour
        _canvas[_cy - y, _cx - x] = _colour
        _canvas[_cy - y, _cx + x] = _colour

    i = 0
    j = outer_radius
    last_fade_amount = 0
    # fade_amount = 0

    max_opaque = 1.0

    while i < j:
        height = math.sqrt(max(outer_radius * outer_radius - i * i, 0))
        fade_amount = max_opaque * (math.ceil(height) - height)

        if fade_amount < last_fade_amount:
            # Opaqueness reset so drop down a row.
            j -= 1
        last_fade_amount = fade_amount

        # We're fading out the current j row, and fading in the next one down.
        if max_opaque - fade_amount > threshold:
            _draw_4point(canvas, center_x, center_y, i, j, colour)
        if fade_amount > threshold:
            _draw_4point(canvas, center_x, center_y, i, j - 1, colour)

        i += 1

    if fill:
        boundary_fill4(canvas, center_x, center_y, colour, colour)


def flood_fill4(canvas, start_x, start_y, old_value, fill_value):
    x, y = start_x, start_y
    if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
        if canvas[y, x] == old_value:
            canvas[y, x] = fill_value
            # Attempt to propagate in each of four directions
            flood_fill4(canvas, x, y - 1, old_value, fill_value)
            flood_fill4(canvas, x, y + 1, old_value, fill_value)
            flood_fill4(canvas, x - 1, y, old_value, fill_value)
            flood_fill4(canvas, x + 1, y, old_value, fill_value)


def boundary_fill4(canvas, start_x, start_y, boundary_value, fill_value):
    x, y = start_x, start_y
    if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
        if canvas[y, x] != boundary_value and canvas[y, x] != fill_value:
            canvas[y, x] = fill_value
            boundary_fill4(canvas, x, y - 1, boundary_value, fill_value)
            boundary_fill4(canvas, x, y + 1, boundary_value, fill_value)
            boundary_fill4(canvas, x - 1, y, boundary_value, fill_value)
            boundary_fill4(canvas, x + 1, y, boundary_value, fill_value)
