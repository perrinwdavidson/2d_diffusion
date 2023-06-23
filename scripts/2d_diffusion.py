# =========================================================
# 2d_diffusion
# ---------------------------------------------------------
# purpose :: an example of 2d diffusion with absorbing bcs
#            and explicit time stepping
# author :: perrin w. davidson
# date :: 18.06.2023
# contact :: perrinwdavidson@gmail.com
# =========================================================
# ---------------------------------------------------------
# first, we want to create a github repository. from our working
# directory root, we first initialize with:
#   $ git init -b main
# where main is the name of the local branch specified with the -b flag.
# next, we want to add all the current files to our index:
#   $ git add .
# now, we will commit to our local branch:
#   $ git commit -m "adding in initial files"
# where we have the message added with -m. next, we want to 
# create a remote branch on our github (where we have the 
# authentication tokens already set up) with:
#   $ gh repo create 2d_diffusion --private --source=. --remote=origin
# where here we are making a github repo named 2d_diffusion
# that is private from the entire directory with a remote branch
# named origin. this should do the trick! note that you might
# need to download the cli for github with:
#   $ brew install gh
# and also maybe homebrew with:
#   $ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# which is a great package manager for mac. follow the prompts with answers:
#   - GitHub.com
#   - HTTPS
#   - Yes
#   - Login with a web browser
# lastly, we will push our directory from the local branch main
# to our remote repository origin/main with:
#   $ git push origin main
# next, before we load the packages, we want to create a virtual
# environment with which to code in. i am using python, for
# which i use anaconda to manage my packages and distributions
# of python. i will create a new environment from a .yml file with:
#   $ conda env create -f src/environment.yml
# where i have specified the environment name and packages needed
# within the .yml file. then, i will activate the environment with:
#   $ conda activate 2d_diffusion
# where 2d_diffusion is the name specified in the .yml file.
# now, i am ready to go.
# ---------------------------------------------------------
# configure -----------------------------------------------
# packages ::
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation

# plotting ::
plt.rc("text", usetex=True)
plt.rcParams.update({'font.size': 16})
plt.rc("font", family="serif")
plt.rc("text.latex", preamble=r"\usepackage{wasysym}")


# functions -----------------------------------------------
def bi_normal(x, y, p):
    """
    purpose :: bivariate normal distribution
    inputs ::
        [1] x, y - coordinates 
        [2] p - (0) mu, mean (1) sigma, standard deviation (2) x_0, center of x (3) y_0, center of y
    outputs ::
        [1] bivariate normal distribution
    """
    # calculate l2 norm ::
    d = np.sqrt(((x - p[2]) ** 2) + ((y - p[3]) ** 2))

    # return distribution ::
    return np.exp(-(((d - p[0]) ** 2) / (2.0 * (p[1] ** 2))))


def calculate(c, p):
    """
    purpose :: employ explicit finite difference method to solve diffusion equation
    inputs ::
        [1] c - initial concentration of tracer
        [2] p - (0) # of time steps (1) length of domain (2) spatial resolution (3) r = (D * dt) / (dx ^ 2)
    outputs ::
        [1] c - final concentration of tracer
    """
    # loop through all dimensions ::
    for k in range(0, p[0]-1, 1):
        for i in range(1, p[1]-1, p[2]):
            for j in range(1, p[1]-1, p[2]):

                # calculate laplacian ::
                diffusion = p[3] * (c[k][i+1][j] + c[k][i-1][j] + c[k][i][j+1] + c[k][i][j-1] - (4 * c[k][i][j]))

                # iterate through time ::
                c[k + 1, i, j] = diffusion + c[k][i][j]

    # return final concentration ::
    return c


def plot_map(c_k, k):
    """
    purpose :: plot map
    inputs ::
        [1] c_k - the map at the time index k
        [2] k - the time index
    outputs ::
        [ ] a plot of the map
    """
    # clear the current plot figure ::
    plt.clf()

    # title and label ::
    plt.title("Temperature at $t = " + str(round(k * delta_t, 1)) + "$ [time]")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")

    # plot and return ::
    plt.pcolormesh(c_k, cmap=plt.cm.summer, vmin=0, vmax=0.5)
    plt.colorbar()
    return plt


def animate(k):
    """
    purpose :: animate plots at time k
    inputs ::
        [1] k - the time index
    outputs ::
        [ ] a plot of the map
    """
    # plot ::
    plot_map(c[k], k)

    # write out ::
    if k % 20 == 0:
        print("Done with time step " + str(k))


# perform main routine ------------------------------------
if __name__ == "__main__":

    # set constants ---------------------------------------
    # calculate parameters ::
    p_calc = [500, 50, 1]
    diffusion_coeff = 2
    delta_t = (p_calc[2] ** 2)/(4 * diffusion_coeff)
    p_calc.append((diffusion_coeff * delta_t) / (p_calc[2] ** 2))

    # bivariate normal distribution parameters ::
    p_dist =  [0.0, 10, (p_calc[1] / 2), (p_calc[1] / 2)]  # mu, sigma, x_0, y_0

    # initialize ------------------------------------------
    # initialize the concentration with grid c(k, i, j)
    c = np.empty((p_calc[0], p_calc[1], p_calc[1]))

    # generate initial concentration ::
    x_grid, y_grid = np.meshgrid(np.linspace(0, p_calc[1], p_calc[1]), 
                                 np.linspace(0, p_calc[1], p_calc[1]))
    c_initial = bi_normal(x_grid, y_grid, p_dist)

    # initialize array ::
    c[0, :, :] = c_initial

    # plot concentration ::
    plt.figure()
    plt.pcolormesh(c_initial, cmap=plt.cm.summer, vmin=0, vmax=0.5)
    plt.colorbar()
    plt.title(r"Initial Temperature Condition")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.show()

    # boundary conditions (perfectly absorbing) -----------
    # set ::
    c_top = 0.0
    c_left = 0.0
    c_bottom = 0.0
    c_right = 0.0

    # place the boundary conditions ::
    c[:, (p_calc[1]-1):, :] = c_top
    c[:, :, :1] = c_left
    c[:, :1, 1:] = c_bottom
    c[:, :, (p_calc[1]-1):] = c_right

    # perform calculation ---------------------------------
    c = calculate(c, p_calc)

    # plot calculation ------------------------------------
    # animate ::
    anim = FuncAnimation(plt.figure(),
                         animate,
                         interval=1,
                         frames=p_calc[0],
                         repeat=True)

    # save ::
    anim.save("plots/diffusion.gif")

# =========================================================
