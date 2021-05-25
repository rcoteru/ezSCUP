
import matplotlib.pyplot as plt
import numpy as np

def plot_2D_vector_domain(
    field:np.ndarray, fname:str = "graph.png", save:bool = True, close:bool = False,
    title:str = "2D Vector Field", axis:str = "z", layer:int = 0):

    label_size = 14
    figure_pad = 2

    X, Y = np.meshgrid(range(field.shape[0]), range(field.shape[1]))
    u, v = field[:,:,layer,0], field[:,:,layer,1]
    
    fig = plt.figure(figsize=[8,8]) 
    ax  = fig.add_subplot(1, 1, 1) 
    plt.tight_layout(pad = figure_pad)
    plt.title(title)

    m = np.mean(np.hypot(u, v))
    q0 = ax.quiver(X, Y, v, u, pivot="mid", width=0.008, headwidth=5, minlength=0, minshaft=3)
    ax.quiverkey(q0, 0.9, 1.03, m, '{:2.1f} º'.format(m), labelpos='E')
    ax.set_xlabel("$x$ (unit cells)", fontsize = label_size)
    ax.set_ylabel("$y$ (unit cells)", fontsize = label_size)
    #ax.invert_yaxis()

    plt.savefig(fname)
    if close:
        plt.close()
