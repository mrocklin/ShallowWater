from shallowwater_base import *

def evolveEuler(eta, u, v, g, dt=dt):
    """
    Evolve state (eta, u, v, g) forward in time using simple Euler method
    x_{n+1} = x_{n} +   dx/dt * d_t

    Returns an generator / infinite list of all states in the evolution

    >>> trajectory = evolveEuler(eta, u, v, g)
    >>> eta, u, v, time = trajectory.next()
    >>> eta, u, v, time = trajectory.next()
    >>> imshow(eta)
    """
    time = 0
    yield eta, u, v, time # return initial conditions as first state in sequence

    while(True):
        eta, u, v = step(eta, u, v, g, dt)
        time += dt

        yield eta, u, v, time

def demo(eta=eta_start, u=u_start, v=v_start, g=g, dt=dt, endTime=.3):
    trajectory = evolveEuler(eta, u, v, g, dt)

    # Figure with initial conditions
    eta, u, v, time = trajectory.next()
    figure(); title('Initial conditions')
    imshow(eta); colorbar()

    # Burn some time
    time = 0
    while(time < endTime):
        _, _, _, time = trajectory.next()

    # Figure after some time has passed
    eta, u, v, time = trajectory.next()
    figure(); title('time=%f'%time)
    imshow(eta); colorbar()
