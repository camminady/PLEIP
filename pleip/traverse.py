from numpy import inf, isclose

def nextcell(pos, velocity, x0, x1, y0, y1):
    """
    Computes into which of the 8 neighboring cells we
    are going to move into.

    Args:
        pos: Particle positions.
        velocity: Particle velocity.
        x0: Start of domain in x-direction.
        x1: End of domain in x-direction.
        y0: Start of domain in y-direction.
        y1: End of domain in y-direction.

    Returns:
        nextx: Either -1 (left), 0 (center), +1 (right) for whether we move horizontally to left, middle, right.
        nexty: Either -1 (left), 0 (center), +1 (right) for whether we move vertically  to left, middle, right.
        dist: The distance that a particle will travel until it leaves the domain.

    Raises:
        ValueError: If the particle is not in the domain.
        ValueError: If the velocity is zero.
    """

    if not (x0 <= pos[0] <= x1) or not (x0 <= pos[0] <= x1):
        raise ValueError('Particle at ({},{}) not inside [{},{}]x[{},{}].'.format(pos[0], pos[1], x0, x1, y0, y1))

    if isclose(velocity[0] ** 2 + velocity[1] ** 2, 0.0):
        raise ValueError('Velocity can not be zero.')

    # are we moving north east, south east, south west or north west
    xdirection = +1 if velocity[0] >= 0 else -1
    ydirection = +1 if velocity[1] >= 0 else -1

    # which face (left right top bottom) will we hit
    # i.e. to which y or x do we need to calculate the distance
    x = x1 if xdirection == +1 else x0
    y = y1 if ydirection == +1 else y0

    # calculate distance to that point considering the velocity
    distx = inf if isclose(velocity[0], 0.0) else abs((x - pos[0]) / velocity[0])
    disty = inf if isclose(velocity[1], 0.0) else abs((y - pos[1]) / velocity[1])

    # check which face we hit first and save the corresponding i/j shift
    # as well as the distance travelled
    if disty < distx:
        nextij = (0, ydirection)
        dist = disty
    elif disty > distx:
        nextij = (xdirection, 0)
        dist = distx
    else:
        nextij = (xdirection, ydirection)
        dist = distx  # = disty

    return nextij[0], nextij[1], dist
