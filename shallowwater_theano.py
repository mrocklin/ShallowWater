from pylab import figure, imshow, title, colorbar
import theano.tensor as T
import theano
from shallowwater_base import *

def make2DTensor(name):
    """
    Wraps T.TensorType for this application

    >>> eta = make2DTensor('eta')
    """
    return T.TensorType(dtype='float64', broadcastable=(False,False))(name)

def evolveTime(g, endTime, dt=dt):
    """
    Creates theano function

        to evolve (eta,u,v) at time zero
        to (eta,u,v) at time endTime

    >>> f = evolveTime(1, .3)
    >>> eta_end, u_end, v_end = f(eta_start, u_start, v_start)
    """

    eta_in = make2DTensor('eta')
    u_in = make2DTensor('u')
    v_in = make2DTensor('v')

    numsteps = int(endTime/dt)

    in_state = (eta_in, u_in, v_in)
    result, updates = theano.scan(
            fn = lambda eta,u,v: step(eta,u,v, g, dt),
            outputs_info = in_state,
            n_steps = numsteps)

    eta_out = result[0][-1]
    u_out = result[1][-1]
    v_out = result[2][-1]

    f = theano.function([eta_in, u_in, v_in], [eta_out, u_out, v_out])
    return f

def demo(eta=eta_start, u=u_start, v=v_start, g=g, dt=dt, endTime=.3):

    # Create time evolution function
    f = evolveTime(1, endTime, dt)

    # Figure with initial conditions
    figure(); title('Initial conditions')
    imshow(eta); colorbar()

    # evolve forward in time
    eta, u, v = f(eta_start, u_start, v_start)

    # Figure after some time has passed
    figure(); title('time=%f'%endTime)
    imshow(eta); colorbar()
