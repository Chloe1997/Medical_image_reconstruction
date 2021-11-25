import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from mpl_toolkits  import  mplot3d


def iDFT_one(x):
    x = np.asarray(x, dtype = "complex_")
    M = x.shape[0]
    #  print(M)
    m = np.arange(M)
    u = m.reshape((M, 1))
    #  print(u.shape)
    Matrix = np.exp(2j * np.pi * m * u / M,dtype = "complex_")
    #  print('M',Matrix.shape)
    #  print('x',x.shape)
    return np.dot(Matrix, x)


def iDFT_two(x):
    shape1 = x.shape[0] # M
    shape2 = x.shape[1] # N

    FT1 = np.zeros((shape1,shape2),dtype = "complex_")
    FT2 = np.zeros((shape1,shape2),dtype = "complex_")

    for col in range(shape2):
        FT2[:,col] = DFT_one(x[:, col])[::-1]

    for row in range(shape1):
        FT1[row, :] = DFT_one(FT2[row, :])[::-1]

        # FT1 = np.asarray(FT1)
    print('FT1', np.shape(FT1))
    print('FT2', np.shape(FT2))

    FT = np.array(FT1,dtype = "complex_")
    print(FT.shape)
    return FT

def DFT_one(x):
    x = np.asarray(x, dtype="complex_")
    M = x.shape[0]
    #  print(M)
    m = np.arange(M)
    u = m.reshape((M, 1))
    #  print(u.shape)
    Matrix = np.exp(-2j * np.pi * m * u / M,dtype = "complex_")
    #  print('M',Matrix.shape)
    #  print('x',x.shape)
    return np.dot(Matrix, x)


def DFT_two(x):
    shape1 = x.shape[0] # M
    shape2 = x.shape[1] # N
    # FT1 = []
    # FT2 = []
    FT1 = np.zeros((shape1,shape2),dtype = "complex_")
    FT2 = np.zeros((shape1,shape2),dtype = "complex_")
    for row in range(shape1):
        # print('shape',DFT_one(x[row,:],shape2).shape)
        # print(np.expand_dims(DFT_one(x[row,:],shape2),axis=1).shape)
        # FT1.append(DFT_one(x[row, :], shape2))
        FT1[row,:] = DFT_one(x[row, :])
    # FT1 = np.asarray(FT1)
    print('FT1', np.shape(FT1))
    for col in range(shape2):
        # FT2.append(DFT_one(FT1[:, col].T, shape1))
        FT2[:,col] = DFT_one(FT1[:, col])

    # FT2 = np.asarray(FT2)
    print('FT2', np.shape(FT2))
    # FT = FT2.T
    # print(FT)
    FT = np.array(FT2,dtype = "complex_")
    print(FT.shape)
    return FT / (shape1 * shape2)

def demo():
    img = cv.imread('./head_phantom_4.png')
    # img = cv.imread('./coin.png')
    shape1 = img.shape[0] # M
    shape2 = img.shape[1] # N
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    f = np.fft.fft2(img)
    magnitude_spectrum2 = 20*np.log(np.abs(f))
    print(np.shape(img))
    FT = DFT_two(img)
    magnitude_spectrum = 20*np.log(np.abs(FT))
    magnitude_spectrum3 = 20*np.log(np.abs(f-FT))



    z = abs(FT)
    print(z.shape)
    IM = iDFT_two(FT).real
    IM2 = np.fft.ifftn(f).real


    FT = np.fft.fftshift(FT)
    print(FT.shape)
    plt.subplot(141),plt.imshow(magnitude_spectrum2, cmap = 'gray')
    plt.title('FT Function'), plt.xticks([]), plt.yticks([])
    plt.subplot(142),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('FT'), plt.xticks([]), plt.yticks([])
    plt.subplot(143),plt.imshow(magnitude_spectrum3, cmap = 'gray')
    plt.title('IFT Function'), plt.xticks([]), plt.yticks([])
    plt.subplot(144),plt.imshow(IM, cmap = 'gray')
    plt.title('IFT'), plt.xticks([]), plt.yticks([])
    plt.show()

# demo()


def input(file_path):
    img = cv.imread(file_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    FT = DFT_two(img)
    magnitude_spectrum = 20 * np.log(np.abs(FT))
    # define figure
    plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('2DFT',bbox_inches='tight',dpi=100)
    plt.show()
    return './2DFT.png'

# print(input('head_phantom_4.png'))

# for i in range(200):
#     plt.ion()
#     img = cv.imread('./coin.png')
#     img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     ft1 = DFT_one(img[i, :])
#     plt.figure()
#     plt.plot(ft1)
#     plt.show()


