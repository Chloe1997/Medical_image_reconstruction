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

    elif (name.lower() == 'test'):
        e = [[0, -.0184, .8740, .6624, 90, -0.8],
            [.22, 0, .3100, .11 , 72,-.2]]
    return e

def phantom(type='shepp-logan'):
    ellipses = _select_phantom(type)
    # define figure
    fig = plt.figure(figsize=(4, 4))
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
        xcenter, ycenter = ellip[0],ellip[1]
        Major, Minor = ellip[2],ellip[3]
        angle = ellip[4]*np.pi/180
        t = np.arange(0, 2 * np.pi, 0.01)
        I = (ellip[5]+max_intensity)*(1/(2*max_intensity))  # Intensity
        i = np.arange(0, Major, 0.01)
        j = np.arange(0, Minor, 0.01)
        x, y = [], []
        for m in i:
            for n in j:
                x.append(np.cos(angle) * (np.cos(t) * m ) - np.sin(angle) * (np.sin(t) * n)+ xcenter)
                y.append(np.sin(angle) * (np.cos(t) * m ) + np.cos(angle) * (np.sin(t) * n)+ ycenter)
        ax.plot(x, y, c=str(I))

    # plt.grid(ls='--')

    plt.xticks([]),plt.yticks([])
    # plt.savefig('head_phantom_1',bbox_inches='tight',dpi=100)
    plt.savefig('head_modified',bbox_inches='tight',dpi=100, pad_inches = 0)
    # plt.savefig('head_shepp',bbox_inches='tight',dpi=100)

    plt.show()

phantom(type='modified shepp-logan')
# phantom(type='shepp-logan')
# phantom(type='test')


