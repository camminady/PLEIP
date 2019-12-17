from numpy import inf, sqrt, zeros, dot


def distance_periodic(point0, point1, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    """
    Computes the periodic distance between two points.
    This function defines the topology of the domain. If we set the distance
    to work in a periodic setting, then everything that uses
    this distance function will automatically use the periodic case.

    Args:
        point0: numpy.array with two entries in [x0,x1[ x [y0,y1[
        point1: numpy.array with two entries in [x0,x1[ x [y0,y1[
        x0: Lower bound of domain in x-direction.
        x1: Upper bound of domain in x-direction.
        y0: Lower bound of domain in y-direction.
        y1: Upper bound of domain in y-direction.

    Returns:
        minimaldistance: The distance between point0 and point1 in a periodic domain.

    Raises:
        ValueError: If x0>=x1 or y0>=y1.
        ValueError: If point0 or point1 is not in the domain.
    """

    if x0 > x1 or y0 > y1:
        raise ValueError('We demand x0<x1 and y0<y1 but got x0={}, x1={}, y0={}, and y1={}'.format(x0, x1, y0, y1))
    if not (x0 <= point0[0] < x1) or not (y0 <= point0[1] < y1):
        raise ValueError('point0=({},{}) is not inside [{},{}[ x [{},{}[.'.format(point0[0], point0[1], x0, x1, y0, y1))
    if not (x0 <= point1[0] < x1) or not (y0 <= point1[1] < y1):
        raise ValueError('point1=({},{}) is not inside [{},{}[ x [{},{}[.'.format(point1[0], point1[1], x0, x1, y0, y1))

    minimaldistance = inf  # Initialize with inf and then find the minimum over all periodic cases.
    width, height = x1 - x0, y1 - y0
    for offsetx in [-width, 0.0, width]:
        for offsety in [-height, 0.0, height]:
            dx = (point0[0] - x0 + offsetx) - (point1[0] - x0)  # Difference in x-dimension.
            dy = (point0[1] - y0 + offsety) - (point1[1] - y0)  # Difference in y-dimension.
            dist = sqrt(dx ** 2 + dy ** 2)
            if dist < minimaldistance: minimaldistance = dist
    return minimaldistance


def customnorm(v, distance=distance_periodic):
    """
    Generalizes the norm two arbitrary distance functions.

    Args:
        v: numpy.array of which we compute the norm.
        distance: The distance function with respect to which we compute the norm.

    Returns:
        distance(v,zeros(v.shape))
    """

    return distance(v, zeros(v.shape))


def distance_pointline(linepoint, linedirection, point, norm=customnorm):
    """
    Computes the minimal distance between a point and a line.
    See: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Vector_formulation

    Args:
        linepoint: A point through which a line passes.
        linedirection: The direction vector that defines the line together with linepoint.
        point: We compute the distance to the line for exactly this point
        distance: A distance function that takes two points and returns their distance.

    Returns:
        minmaldistance: The minimal distance between a point and a line.
        s: The value for which distance(linepoint+linedirection*s - point)=minimaldistance.
    """
    speed = sqrt(linedirection[0] ** 2 + linedirection[1] ** 2)
    normalvector = linedirection / speed
    minmaldistance = norm((linepoint - point) - normalvector * (dot(linepoint - point, normalvector)))
    s = sqrt((linepoint - point)[0] ** 2 + (linepoint - point)[1] ** 2 - minmaldistance ** 2)
    return minmaldistance, s
