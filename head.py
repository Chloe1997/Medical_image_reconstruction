import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt


def _select_phantom(name):
    if (name.lower() == 'shepp-logan'):
        e = [[0, 0, .92, .69, 90, 2],
            [0, -.0184, .8740, .6624, 90, -.98],
            [.22, 0, .3100, .11 , 72,-.02],
            [-.22, 0,.41, .16, 108,-.02],
            [0, .35, .2500, .2100,  0,.01],
            [ 0, .1, .0460, .0460, 0,.01],
            [ 0, -.1, .0460,.0460, 0,.02],
            [-.08, -.605,  .0230,.0460,  0,.01],
            [ 0, -.606,  .0230,.0230, 0,.01],
            [.06, -.605,  .0460,.0230, 90,.01]]
    elif (name.lower() == 'modified shepp-logan'):
        e = [[0, 0, .92, .69, 90, 1],
            [0, -.0184, .8740, .6624, 90, -0.8],
            [.22, 0, .3100, .11 , 72,-.2],
            [-.22, 0,.41, .16, 108,-.2],
            [0, .35, .2500, .2100,  0,.1],
            [ 0, .1, .0460, .0460, 0,.1],
            [ 0, -.1, .0460,.0460, 0,.2],
            [-.08, -.605,  .0230,.0460,  0,.1],
            [ 0, -.606,  .0230,.0230, 0,.1],
            [.06, -.605,  .0460,.0230, 90,.1]]
    return e

def phantom(type='shepp-logan'):
    ellipses = _select_phantom(type)
    p = []
    # define figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xbound(-1, 1)
    ax.set_ybound(-1, 1)
    ax.set_xticks(np.arange(-1, 1.2, 0.2))
    ax.set_yticks(np.arange(-1, 1.2, 0.2))

    ax.set_facecolor((0, 0, 0))
    # ax.set_axis_off()

    #calculate intensity
    e = np.array(ellipses)
    max_intensity = max(abs(e[:,5]))
    # print(max_intensity)

    for ellip in ellipses:
        # xcenter, ycenter = ellip[0]+(1/n),ellip[1]+(1/n)
        xcenter, ycenter = ellip[0],ellip[1]

        print( xcenter, ycenter)
        width, height = ellip[2]*2,ellip[3]*2
        angle = ellip[4]
        # I = (ellip[5]+2)*0.25  # Intensity
        I = (ellip[5]+max_intensity)*(1/(2*max_intensity))  # Intensity
        e = patches.Ellipse((xcenter, ycenter), width, height,angle=angle, color=str(I), fill=True)
        p.append(e)
    [ax.add_patch(i) for i in p]
    plt.grid(ls='--')
    # cm = plt.cm.get_cmap('gray')
    # xy = range(2)
    # z = xy
    # sc = plt.scatter(xy, xy, c=z, vmin=-1, vmax=1, s=2, cmap=cm)
    # plt.colorbar(sc)
    plt.xticks([]),plt.yticks([])
    plt.savefig('head_phantom_1',bbox_inches='tight',dpi=100)
    # plt.savefig('head_modified',bbox_inches='tight',dpi=100)
    # plt.savefig('head_shepp',bbox_inches='tight',dpi=100)

    plt.show()

# phantom(type='modified shepp-logan')
phantom(type='shepp-logan')

