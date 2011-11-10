from numpy import *
from pylab import figure, imshow, title, colorbar
import theano.tensor as T
import theano



# Initial Conditions
n = 100
u_start = zeros((n,n)) # velocity in x direction - still water
v_start = zeros((n,n)) # velocity in y direction - still water

# eta (like height) will be uniform with a perturbation in the center
eta_start = ones((n,n)) # pressure deviation (like height)
x,y = mgrid[:n,:n]
droplet_x, droplet_y = n/2, n/2
rr = (x-droplet_x)**2 + (y-droplet_y)**2
eta_start[rr<10**2] = 1.1 # add a perturbation in pressure surface

# Parameters describing simulation
box_size = 1.
grid_spacing =  1.0*box_size / n
g = 1. # Gravity
dt = grid_spacing / 100.

def make2DTensor(name):
    """
    Wraps T.TensorType for this application

    >>> eta = make2DTensor('eta')
    """
    return T.TensorType(dtype='float64', broadcastable=(False,False))(name)

def roll(x, shift, axis):
    """
    A reinvention of numpy.roll using the Theano RollOp
    """
    return T.RollOp(shift, axis)(x)

def spatial_derivative(A, axis=0):
    """
    Compute derivative of array A using balanced finite differences
    Axis specifies direction of spatial derivative (d/dx or d/dy)

    dA[i] =  A[i+1] - A[i-1]   / 2
    ... or with grid spacing included ...
    dA[i]/dx =  A[i+1] - A[i-1]   / 2dx

    Used By:
        d_dx
        d_dy
    """
    return (roll(A, -1, axis) - roll(A, 1, axis)) / (grid_spacing*2.)

def d_dx(A):
    return spatial_derivative(A,1)
def d_dy(A):
    return spatial_derivative(A,0)


def d_dt(eta, u, v, g, b=0):
    """
    http://en.wikipedia.org/wiki/Shallow_water_equations#Non-conservative_form
    """
#    for x in [eta, u, v]: # type check
#        assert isinstance(x, ndarray) and not isinstance(x, matrix)

    du_dt = -g*d_dx(eta) - b*u
    dv_dt = -g*d_dy(eta) - b*v

    H = 0#eta.mean() - our definition of eta includes this term
    deta_dt = -d_dx(u * (H+eta)) - d_dy(v * (H+eta))

    return deta_dt, du_dt, dv_dt

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

def step(eta, u, v, g, dt=dt):
    """
    Step forward eta, u, v one step in time of duration dt

    See Also:
        d_dt
    """
    deta_dt, du_dt, dv_dt = d_dt(eta, u, v, g)

    eta = eta + deta_dt * dt
    u = u + du_dt * dt
    v = v + dv_dt * dt

    return (eta, u, v)

def demo(eta=eta_start, u=u_start, v=v_start, g=g, dt=dt, endTime=.3):

    # Create time evolution function
    f = evolveTime(1, endTime, dt)

    # Figure with initial conditions
    figure(); title('Initial conditions')
    imshow(eta); colorbar()

    # evolve forward in time
    eta, u, v = f(eta, u, v)

    # Figure after some time has passed
    figure(); title('time=%f'%endTime)
    imshow(eta); colorbar()
