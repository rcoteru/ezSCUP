
import matplotlib.pyplot as plt

def plot_vector(name, u, v, t="", save=True):

    """ Plots a vector map. """

    plt.figure(name, figsize=[6,6])
    plt.quiver(u,v, pivot="mid")
    if t != "":
        plt.title(t)
    plt.xlabel("x (unit cells)", fontsize=12)
    plt.ylabel("y (unit cells)", fontsize=12)
    plt.tight_layout()
    if save:
        plt.savefig(name)
    plt.draw()
    plt.close()

def plot_vectors(name, u1, v1, u2, v2, t1="", t2="", save=True):

    """ Plots two vector maps side by side. """

    fig, axs = plt.subplots(1,2, figsize=[12,6], sharey=True)
    plt.tight_layout()
    fig.canvas.set_window_title(name) 
    
    axs[0].quiver(u1,v1, pivot="mid")
    if t1 != "":
        axs[0].set_title(t1)
    axs[0].set_xlabel("x (unit cells)", fontsize=12)
    axs[0].set_ylabel("y (unit cells)", fontsize=12)
    axs[0].invert_yaxis()

    axs[1].quiver(u2,v2, pivot="mid")
    if t2 != "":
        axs[1].set_title(t2)
    axs[1].set_xlabel("x (unit cells)", fontsize=12)
    axs[1].invert_yaxis()

    if save:
        plt.savefig(name)
    plt.draw()
    plt.close()


def plot_heatmap(name, a, t="Title", save=True):

    """ Plots a heatmap. """

    plt.figure(name, figsize=[6,6])
    plt.imshow(a, cmap="seismic", interpolation="nearest")
    plt.colorbar()
    plt.title(t)
    plt.xlabel("x (unit cells)", fontsize=12)
    plt.ylabel("y (unit cells)", fontsize=12)
    plt.tight_layout()
    if save:
        plt.savefig(name)
    plt.draw()
    plt.close()

def plot_heatmaps(name, a, b, t1="Title1", t2="Title2", save=True):

    """ Plots two heatmaps side by side. """

    fig, axs = plt.subplots(1,2, figsize=[12,6], sharey=True)
    plt.tight_layout()
    fig.canvas.set_window_title(name) 
    
    im1 = axs[0].imshow(a, cmap="seismic", interpolation="nearest")
    fig.colorbar(im1, ax=axs[0])
    axs[0].set_title(t1)
    axs[0].set_xlabel("x (unit cells)", fontsize=12)
    axs[0].set_ylabel("y (unit cells)", fontsize=12)
    axs[0].invert_yaxis()

    im2 = axs[1].imshow(b, cmap="seismic", interpolation="nearest")
    fig.colorbar(im2, ax=axs[1])
    axs[1].set_title(t2)
    axs[1].set_xlabel("x (unit cells)", fontsize=12)
    axs[1].invert_yaxis()
    
    if save:
        plt.savefig(name)
    plt.draw()
    plt.close()
