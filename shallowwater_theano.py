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

#    eta,u,v = eta_in, u_in, v_in
#    for i in xrange(numsteps):
#        eta,u,v = step(eta,u,v,g,dt)
#    eta_out, u_out, v_out = eta,u,v
#    eta_out.name = 'eta_out'
#    u_out.name = 'u_out'
#    v_out.name = 'v_out'

    in_state = (eta_in, u_in, v_in)
    result, updates = theano.scan(
            fn = lambda eta,u,v: step(eta,u,v, g, dt),
            outputs_info = in_state,
            n_steps = numsteps, name='timeEvolutionScanOp')

    eta_out = result[0][-1]; eta_out.name = 'eta_out'
    u_out = result[1][-1]; u_out.name = 'u_out'
    v_out = result[2][-1]; v_out.name = 'v_out'

    return eta_in, u_in, v_in, eta_out, u_out, v_out

def build_f(g, endTime, dt=dt, **kwargs):
    eta_in, u_in, v_in, eta_out, u_out, v_out = evolveTime(g,endTime, dt)

    f = theano.function([eta_in, u_in, v_in], [eta_out, u_out, v_out], **kwargs)

    return f

def build_fp(g, endTime, dt=dt, **kwargs):
    eta_in, u_in, v_in, eta_out, u_out, v_out = evolveTime(g,endTime, dt)

    deta_in = make2DTensor('deta')
    du_in = make2DTensor('du')
    dv_in = make2DTensor('dv')

    deta_out, du_out, dv_out = T.Rop((eta_out, u_out, v_out),
            (eta_in, u_in, v_in), (deta_in, du_in, dv_in))

    fp = theano.function([eta_in, u_in, v_in, deta_in, du_in, dv_in],
            [deta_out, du_out, dv_out], **kwargs)

    return fp

def build_fp_bulk(g, endTime, n, dt=dt):
    # the normal computation of eta_out, u_out, v_out = f(eta_in, u_in, v_in)
    eta_in, u_in, v_in, eta_out, u_out, v_out = evolveTime(g,endTime, dt)

    make3DTensor = lambda name : T.TensorType(dtype='float64',
            broadcastable=(False,False,False))(name)

    # Input stacks to hold of delta x
    deta_in_stack = make3DTensor('deta_in_stack')
    du_in_stack = make3DTensor('du_in_stack')
    dv_in_stack = make3DTensor('dv_in_stack')
    deta_outs = []
    du_outs = []
    dv_outs = []

    for i in range(n):
        # Grab slices off of the input stack
        deta_in = deta_in_stack[i]; deta_in.name='deta_in_%d'%i
        du_in = du_in_stack[i]; du_in.name='du_in_%d'%i
        dv_in = dv_in_stack[i]; dv_in.name='dv_in_%d'%i
        # Evolve them with the R_op
        deta_out, du_out, dv_out = T.Rop((eta_out, u_out, v_out),
                (eta_in, u_in, v_in), (deta_in, du_in, dv_in))

        # Add results to a list of outputs
        deta_outs.append(deta_out); deta_out.name = 'deta_out_%d'%i
        du_outs.append(du_out); du_out.name = 'du_out_%d'%i
        dv_outs.append(dv_out); dv_out.name = 'dv_out_%d'%i

    # Stack list of outputs into an output array
    deta_out_stack = T.join(0, deta_outs); deta_out_stack.name='deta_out_stack'
    du_out_stack = T.join(0, du_outs); du_out_stack.name='du_out_stack'
    dv_out_stack = T.join(0, dv_outs); dv_out_stack.name='dv_out_stack'

    # Create theano function x_in, dx_in_stack => dx_out_stack
    fp_bulk = theano.function([eta_in, u_in, v_in,
        deta_in_stack, du_in_stack, dv_in_stack],
        [deta_out_stack, du_out_stack, dv_out_stack])

    return fp_bulk

def demo(eta=eta_start, u=u_start, v=v_start, g=g, dt=dt, endTime=.3):

    # Create time evolution function
    f = build_f(1, endTime, dt=dt)

    # Figure with initial conditions
    figure(); title('Initial conditions')
    imshow(eta); colorbar()

    # evolve forward in time
    eta, u, v = f(eta, u, v)

    # Figure after some time has passed
    figure(); title('time=%f'%endTime)
    imshow(eta); colorbar()

def demo_bulk(endTime = .3, n=3):
    fp = build_fp(1, endTime)
    fp_bulk = build_fp_bulk(1, endTime, n)
    # function takes 2d-array => 3d-array by stacking input n times
    bcast = lambda x: x[None, ...] * ones((n,))[:,None,None]

    #
    de_out, du_out, dv_out = fp(eta_start,u_start,v_start, deta_start,
            du_start, dv_start)
    de_out_stack, du_out_stack, dv_out_stack = fp_bulk(eta_start,u_start,
            v_start, bcast(deta_start), bcast(du_start), bcast(dv_start))

    # Make sure that the computations are identical
    assert (de_out == de_out_stack[1]).all()
