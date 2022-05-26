'''
==========
Jigsaw plot for Structured SC,
Bar plot to show activation strength
Available functions:
activity 
==========
'''

from utils import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import animation 
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_feedforward_weights(Phi, fpath, append = None, suffix = "receptive_field"):
    emb_dim, num_units = Phi.shape
    imsz = int(np.sqrt(emb_dim))
    assert imsz**2 == emb_dim
    sz = int(np.sqrt(num_units))
    assert sz**2 == num_units
    fig = plt.figure(figsize=(sz, sz))
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    for i in range(sz**2):
        plt.subplot(sz, sz, i+1)
        plt.imshow(np.reshape(Phi[:, i], (imsz, imsz)), cmap="gray")
        plt.axis("off")

    if append is None:
        append = ''

    fig.suptitle(fpath[10:].replace('/', ' ').strip())
    plt.subplots_adjust(top=0.9)
    plt.savefig(fpath + '{}{}.pdf'.format(suffix, append))
    plt.close()

def animateSixFlags(lif_model, step):
    block1 = np.stack(lif_model.voltages, axis=0)
    block2 = np.stack(lif_model.stimuli, axis=0)
    block3 = np.stack(lif_model.exc_currents, axis=0)
    block4 = np.stack(lif_model.inh_currents, axis=0)

    figure1 = np.array(lif_model.firing_rate)
    figure2 = np.array(lif_model.betas)

    l = lif_model.side_len
    n = block1.shape[0]
    s = lif_model.steps_per_plot
    lr = lif_model.lr * 1000 # convert unit ms

    titles = [
        "Voltage",
        "Stimulus",
        "Excitatory Current",
        "Inhibitory Current",
        "Firing Rate",
        "E/I Current Ratio"
    ]

    blocks = [block1, block2, block3, block4]
    figures = [figure1, figure2]

    x = np.linspace(-1,1,l)
    y = np.linspace(-1,1,l)
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()

    fig = plt.figure(constrained_layout = False, figsize=(10,12))
    spec = fig.add_gridspec(nrows = 6, ncols = 4)

    ax1 = fig.add_subplot(spec[:2,:2])
    ax2 = fig.add_subplot(spec[:2,2:])
    ax3 = fig.add_subplot(spec[2:4,:2])
    ax4 = fig.add_subplot(spec[2:4,2:])
    ax5 = fig.add_subplot(spec[4,:])
    ax6 = fig.add_subplot(spec[5,:])

    axes = [ax1,ax2,ax3,ax4,ax5,ax6]

    variables = []

    # Set up the blocks
    for i in range(4):
        ax = axes[i]
        scat = ax.scatter(xx, yy, c=np.zeros(shape=(l**2,)), \
            vmin=blocks[i].min(), vmax=blocks[i].max(), s=80000/l**2, cmap='RdYlBu_r')
        ax.set_title(titles[i])
        ax.set_xlim((-1,1))
        ax.set_ylim((-1,1))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        variables.append(scat)
        # Add colorbar
        divider = make_axes_locatable(ax)
        loc = "left" if i%2 == 0 else "right"
        cax = divider.append_axes(loc, size ="5%", pad=0.05)
        fig.colorbar(scat, cax=cax)
        cax.yaxis.set_ticks_position(loc)
        cax.ticklabel_format(style="sci", scilimits=(-2,2))

    for i, figure in enumerate(figures):
        ax = axes[4+i]
        ax.plot(figure)
        ax.set_title(titles[i+4])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "%d" % (lr*x)))
        vline = ax.axvline(0, c = "gray")
        variables.append(vline)
        # set title, x axes stuff
    ax5.get_xaxis().set_visible(False)
    ax6.set_xlabel("Time Elapsed (ms)")

    time_text = ax1.text(0.5, 7.5, '', transform=ax.transAxes, color = "black", \
        horizontalalignment='center', verticalalignment='bottom', \
            fontsize = 'medium', \
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    
    variables.append(time_text)
    

    def init():
        for i in range(4):
            scat = variables[i]
            scat.set_array([])
        for i in range(4,6):
            vline = variables[i]
            vline.set_xdata([])
        variables[-1].set_text("%d ms"  % 0)
        return variables
        
    def animate(i):
        for j in range(4):
            scat = variables[j]
            scat.set_array(blocks[j][i].flatten())
        for j in range(4,6):
            vline = variables[j]
            vline.set_xdata(s*i)
        variables[-1].set_text("%d ms"  % (lr*s*i))
        return variables

    ani = animation.FuncAnimation(fig, animate, range(0, n), 
        interval=20, init_func=init, blit = True)
    name = '/movie-{:d}-{:d}.mp4'.format(lif_model.max_a_fit, lif_model.seed)
    ani.save(mkdir(lif_model.fpath + "step_" + str(step) + "/") + name, writer='ffmpeg')
    plt.close(fig)


def animateSixFlags2(lif_model, fpath, aniname = "test.mp4"):
    """
    Same as animateSixFlags except designed for simplified LIF model w/o excitation so that it is more like Zylberberg's 
    """
    block1 = np.stack(lif_model.voltages, axis=0)
    block2 = np.stack(lif_model.stimuli, axis=0)
    block3 = np.stack(lif_model.firings, axis=0)
    block4 = np.stack(lif_model.post_cond, axis=0)

    figure1 = np.array(lif_model.firing_rate)
    figure2 = np.array(lif_model.neuron_voltage)

    l = int(np.sqrt(lif_model.num_units))
    n = block1.shape[0]
    print(n)
    s = 1
    lr = lif_model.TUNIT * 1000 # convert unit ms

    titles = [
        "Voltage",
        "Stimulus",
        "Cumulative Firing",
        "Post-conductance",
        "Firing Rate",
        "Potential of 1st neuron"
    ]

    blocks = [block1, block2, block3, block4]
    figures = [figure1, figure2]

    x = np.linspace(-1,1,l)
    y = np.linspace(-1,1,l)
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()

    fig = plt.figure(constrained_layout = False, figsize=(10,12))
    spec = fig.add_gridspec(nrows = 6, ncols = 4)

    ax1 = fig.add_subplot(spec[:2,:2])
    ax2 = fig.add_subplot(spec[:2,2:])
    ax3 = fig.add_subplot(spec[2:4,:2])
    ax4 = fig.add_subplot(spec[2:4,2:])
    ax5 = fig.add_subplot(spec[4,:])
    ax6 = fig.add_subplot(spec[5,:])

    axes = [ax1,ax2,ax3,ax4,ax5,ax6]

    variables = []

    # Set up the blocks
    for i in range(4):
        ax = axes[i]
        scat = ax.scatter(xx, yy, c=np.zeros(shape=(l**2,)), \
            vmin=blocks[i].min(), vmax=blocks[i].max(), s=80000/l**2, cmap='RdYlBu_r')
        ax.set_title(titles[i])
        ax.set_xlim((-1,1))
        ax.set_ylim((-1,1))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        variables.append(scat)
        # Add colorbar
        divider = make_axes_locatable(ax)
        loc = "left" if i%2 == 0 else "right"
        cax = divider.append_axes(loc, size ="5%", pad=0.05)
        fig.colorbar(scat, cax=cax)
        cax.yaxis.set_ticks_position(loc)
        cax.ticklabel_format(style="sci", scilimits=(-2,2))

    for i, figure in enumerate(figures):
        ax = axes[4+i]
        ax.plot(figure)
        ax.set_title(titles[i+4])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "%d" % (lr*x)))
        vline = ax.axvline(0, c = "gray")
        variables.append(vline)
        # set title, x axes stuff
    ax5.get_xaxis().set_visible(False)
    ax6.set_xlabel("Time Elapsed (ms)")

    time_text = ax1.text(0.5, 7.5, '', transform=ax.transAxes, color = "black", \
        horizontalalignment='center', verticalalignment='bottom', \
            fontsize = 'medium', \
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    
    variables.append(time_text)
    

    def init():
        for i in range(4):
            scat = variables[i]
            scat.set_array([])
        for i in range(4,6):
            vline = variables[i]
            vline.set_xdata([])
        variables[-1].set_text("%d ms"  % 0)
        return variables
        
    def animate(i):
        for j in range(4):
            scat = variables[j]
            scat.set_array(blocks[j][i].flatten())
        for j in range(4,6):
            vline = variables[j]
            vline.set_xdata(s*i)
        variables[-1].set_text("%d ms"  % (lr*s*i))
        return variables

    ani = animation.FuncAnimation(fig, animate, range(0, n), 
        interval=20, init_func=init, blit = True)
    # name = '/movie-{:d}-{:d}.mp4'.format(lif_model.max_a_fit, lif_model.seed)
    ani.save(fpath + aniname, writer='ffmpeg')
    # fig.savefig("test.jpg")
    plt.close(fig)


def vis_error(error, fpath, step = None):

    mkdir(fpath)

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1, ncols = 4, \
        figsize = (16,4))
    ax0.plot(error[:,0])
    ax0.set_title("averaged firing counts")
    ax1.plot(error[:,1])
    ax1.set_title("firing rate (l0 norm)")
    ax2.plot(error[:,2])
    ax2.set_title("reconstruction error")
    ax3.plot(error[:,3])
    ax3.set_title("avg words_hat norm")

    if step is None:
        s = ''
    else:
        s = '_' + str(step)

    fig.suptitle(fpath[10:].replace('/', ' ').strip())
    fig.savefig(fpath + 'errors' + s + '.png')
    plt.close()

def animateFourFlags(stimulus, firings, potentials, fpath = "./", aniname = "test.mp4", VT = 0, **kwargs):
    '''
    stimulus: ndarray of shape (num_units)
    firings: ndarray of shape (num_euler_steps, num_units)
    potentials: same as firings
    '''
    activations = np.cumsum(firings, axis = 0)
    data = [stimulus, potentials, activations, firings]

    n_steps, n_units = firings.shape
    sidelen = int(np.sqrt(n_units))

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows = 2, ncols = 2)
    variables = []

    for ax in [ax0, ax1, ax2, ax3]:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    variables = []
    # Time text
    time_text = ax0.text(0.5, 0.8, '', transform=fig.transFigure, color = "black", \
        horizontalalignment='center', verticalalignment='bottom', \
            fontsize = 'medium', \
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    variables.append(time_text)

    # Stimulus
    ax0.imshow(stimulus.reshape(sidelen, sidelen), cmap='RdYlBu_r')
    ax0.set_title("Stimulus")

    
    # Potential
    im = ax1.imshow(np.zeros((sidelen, sidelen)), cmap = 'RdYlBu_r', \
        vmax = VT, vmin = -0.07, interpolation='gaussian')
    variables.append(im)    
    ax1.set_title("Potential")

    # Activation
    act = activations[-1]
    actMax = act.max(); actl0 = np.mean(act > 0); actl1 = np.mean(act)
    im = ax2.imshow(np.zeros((sidelen, sidelen)), cmap = 'RdYlBu_r', \
        vmax = actMax, vmin = 0)
    variables.append(im)    
    ax2.set_title("max={}, l0={:.3f}, l1={:.3f}".format(actMax, actl0, actl1))

    # Firing 
    im = ax3.imshow(np.zeros((sidelen, sidelen)), cmap = 'RdYlBu_r', \
            vmax = firings.max(), vmin = 0)
    variables.append(im)

    def animate(i):
        variables[0].set_text("%d euler steps"  % i)
        for j in range(1,4):
            im = variables[j]
            dat = data[j][i]
            im.set_data(dat.reshape((sidelen,sidelen)))
        return variables

    ani = animation.FuncAnimation(fig, animate, range(1, n_steps), 
        interval=20, blit = False)
    # aniname = kwargs.get("aniname.mp4", "firCompare.mp4")
    ani.save(fpath + aniname, writer = 'ffmpeg')
    plt.close(fig)
    
def compareFiring(acts, fpath, aniname = "test.mp4", **kwargs):
    '''Compare the firing for two models in one learn step
    acts : a list of ndarray of shape (num_euler_steps, num_units)
    names : a list of model name for the first/second acts
    (deprecated) share_scale(boolean): whether the comparison shares the scale
    '''

    n_models = len(acts)
    if "names" in kwargs:
        names = kwargs["names"]
        assert len(names) == (acts)
    else:
        names = list(map(str, range(n_models)))
    BINS = kwargs.get("bins", 10)

    n_steps, n_units = acts[0].shape
    sidelen = int(np.sqrt(n_units))

    # As we are only comparing the firing dynamic and the acts accumulations
    fig, axes = plt.subplots(nrows = 2, ncols = n_models)
    
    variables = []

    # Set up the blocks
    for m in range(n_models):
        # First plot: firing at each euler step
        ax = axes[0,m]
        ax.set_title(names[m]) 
        im = ax.imshow(np.zeros((sidelen, sidelen)), cmap =plt.cm.gray, \
            vmax = 1, vmin = 0)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        variables.append(im)
        # Second plot: histgram of acts
        ax = axes[1, m]
        _, _, bar_container = ax.hist(acts[m].sum(axis = 0))
        # variables.append(bar_container)

    time_text = axes[0,0].text(0.5, 0.8, '', transform=fig.transFigure, color = "black", \
        horizontalalignment='center', verticalalignment='bottom', \
            fontsize = 'medium', \
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    variables.append(time_text)
    print("lenvar", len(variables))

    def init():
        for m in range(n_models):
            data = acts[m][0,:]
            # First plot
            im = variables[m]
            im.set_data(data.reshape((sidelen, sidelen)))
            variables[-1].set_text("%d euler steps"  % 0)
        return variables
    
    def animate(i):
        iterates = []
        for m in range(n_models):
            data = acts[m][i,:]
            # First plot
            im = variables[m]
            im.set_data(data.reshape((sidelen, sidelen)))
            # Second plot
            variables[-1].set_text("%d euler steps"  % i)
        return variables
    
    ani = animation.FuncAnimation(fig, animate, range(1, n_steps), 
        interval=20, init_func=init, blit = False)
    # aniname = kwargs.get("aniname.mp4", "firCompare.mp4")
    ani.save(fpath + aniname, writer = 'ffmpeg')
    plt.close(fig)