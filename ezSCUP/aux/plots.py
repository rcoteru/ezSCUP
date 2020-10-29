
import matplotlib.pyplot as plt

def plot_vector_map(name, u, v):

    plt.figure(name, figsize=[6,6])
    plt.quiver(u,v, pivot="mid")
    plt.xlabel("x (unit cells)", fontsize=12)
    plt.ylabel("y (unit cells)", fontsize=12)
    plt.tight_layout()
    plt.savefig(name)
    plt.draw()

def plot_heatmap(name, a):

    plt.figure(name, figsize=[6,6])
    plt.imshow(a, cmap="seismic", interpolation="nearest")
    plt.colorbar()
    plt.xlabel("x (unit cells)", fontsize=12)
    plt.ylabel("y (unit cells)", fontsize=12)
    plt.tight_layout()
    plt.savefig(name)
    plt.draw()

def plot_heatmaps(name, a, b):

    fig, axs = plt.subplots(1,2, figsize=[12,6], sharey=True)
    plt.tight_layout()
    fig.canvas.set_window_title(name) 
    
    im1 = axs[0].imshow(a, cmap="seismic", interpolation="nearest")
    fig.colorbar(im1, ax=axs[0])
    axs[0].set_xlabel("x (unit cells)", fontsize=12)
    axs[0].set_ylabel("y (unit cells)", fontsize=12)

    im2 = axs[1].imshow(b, cmap="seismic", interpolation="nearest")
    fig.colorbar(im2, ax=axs[1])
    axs[1].set_xlabel("x (unit cells)", fontsize=12)
    
    plt.savefig(name)
    plt.draw()
