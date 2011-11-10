from pylab import figure, imshow, title, colorbar
import theano.tensor as T
import theano
from shallowwater_base import *

deta_start = np.zeros_like(eta_start)
du_start = np.zeros_like(u_start)
dv_start = np.zeros_like(v_start)

deta_start[30,30] = 1

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

    return eta_in, u_in, v_in, eta_out, u_out, v_out

def build_f(g, endTime, dt=dt):
    eta_in, u_in, v_in, eta_out, u_out, v_out = evolveTime(g,endTime, dt)

    f = theano.function([eta_in, u_in, v_in], [eta_out, u_out, v_out])

    return f

def build_fp(g, endTime, dt=dt):
    eta_in, u_in, v_in, eta_out, u_out, v_out = evolveTime(g,endTime, dt)

    deta_in = make2DTensor('deta')
    du_in = make2DTensor('du')
    dv_in = make2DTensor('dv')

    deta_out, du_out, dv_out = T.Rop((eta_out, u_out, v_out),
            (eta_in, u_in, v_in), (deta_in, du_in, dv_in))

    fp = theano.function([eta_in, u_in, v_in, deta_in, du_in, dv_in],
            [deta_out, du_out, dv_out])

    return fp

def build_fp_bulk(g, endTime, n, dt=dt):
    eta_in, u_in, v_in, eta_out, u_out, v_out = evolveTime(g,endTime, dt)

    deta_ins = [make2DTensor('deta_%d'%i) for i in xrange(n)]
    du_ins = [make2DTensor('du_%d'%i) for i in xrange(n)]
    dv_ins = [make2DTensor('dv_%d'%i) for i in xrange(n)]

    deta_outs, du_outs, dv_outs = zip(*[
            T.Rop((eta_out, u_out, v_out), (eta_in, u_in, v_in), (de,du,dv))
            for de,du,dv in zip(deta_ins, du_ins, dv_ins)])

    fp = theano.function([eta_in, u_in, v_in, deta_ins, du_ins, dv_ins],
            [deta_outs, du_outs, dv_outs])

    return fp

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
