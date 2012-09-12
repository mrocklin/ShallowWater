import numpy as np
import theano.tensor as T
from pylab import figure, imshow, title, colorbar

# Initial Conditions
n = 100
u_start = np.zeros((n,n)) # velocity in x direction - still water
v_start = np.zeros((n,n)) # velocity in y direction - still water

# eta (like height) will be uniform with a perturbation in the center
eta_start = np.ones((n,n)) # pressure deviation (like height)
x,y = np.mgrid[:n,:n]
droplet_x, droplet_y = n/2, n/2
rr = (x-droplet_x)**2 + (y-droplet_y)**2
eta_start[rr<10**2] = 1.1 # add a perturbation in pressure surface

# Parameters describing simulation
box_size = 1.
grid_spacing =  1.0*box_size / n
g = 1. # Gravity
dt = grid_spacing / 100.


def roll(x, shift, axis):
    """
    A numpy-theano agnostic version of the numpy.roll operator
    calls either numpy.roll or theano.tensor.roll depending on class

    See numpy.roll for usage
    """
    if isinstance(x, np.ndarray):
        return np.roll(x, shift, axis)
    if isinstance(x, T.basic.TensorVariable):
        return T.roll(x, shift, axis)
    raise NotImplementedError()

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
    du_dt = -g*d_dx(eta) - b*u
    dv_dt = -g*d_dy(eta) - b*v

    H = 0#eta.mean() - our definition of eta includes this term
    deta_dt = -d_dx(u * (H+eta)) - d_dy(v * (H+eta))

    return deta_dt, du_dt, dv_dt

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
