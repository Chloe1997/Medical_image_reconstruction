import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# def INDFT_one(x,M):
#      x = np.asarray(x, dtype=float)
#     #  M = x.shape[id]
#     #  print(M)
#      m = np.arange(M)
#      u = m.reshape((M,1))
#     #  print(u.shape)
#      Matrix = np.exp(-2j * np.pi * m * u )
#     #  print('M',Matrix.shape)
#     #  print('x',x.shape)
#      return np.dot(Matrix, x)
# def INDFT_two(x):
#     shape1 = x.shape[0]
#     shape2 = x.shape[1]
#     FT1 = []
#     FT2 = []
#     for row in range(shape1):
#         # print('shape',DFT_one(x[row,:],shape2).shape)
#         # print(np.expand_dims(DFT_one(x[row,:],shape2),axis=1).shape)
#         FT1.append(INDFT_one(x[row,:],shape2))
#     FT1 = np.asarray(FT1)
#     print('FT1',np.shape(FT1))
#     for col in range(shape2):
#         FT2.append(INDFT_one(FT1[:,col].T,shape1))
#     FT2 = np.asarray(FT2)
#     print('FT2',np.shape(FT2))
#     FT = FT2.T
#     # print(FT)
#     print(FT.shape)
#     return FT/(shape1*shape2)

def iDFT_one(x, M):
    x = np.asarray(x, dtype = "complex_")
    #  M = x.shape[id]
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
        FT2[:,col] = DFT_one(x[:, col], shape1)[::-1]

    for row in range(shape1):
        FT1[row, :] = DFT_one(FT2[row, :], shape2)[::-1]

        # FT1 = np.asarray(FT1)
    print('FT1', np.shape(FT1))
    print('FT2', np.shape(FT2))

    FT = np.array(FT1,dtype = "complex_")
    print(FT.shape)
    return FT

def DFT_one(x, M):
    x = np.asarray(x, dtype="complex_")
    m = np.arange(M)
    u = m.reshape((M, 1))
    Matrix = np.exp(-2j * np.pi * m * u / M , dtype = "complex_")
    return np.dot(Matrix, x)


def DFT_two(x):
    shape1 = x.shape[0] # M
    shape2 = x.shape[1] # N
    # FT1 = []
    # FT2 = []
    FT1 = np.zeros((shape1,shape2),dtype = "complex_")
    FT2 = np.zeros((shape1,shape2),dtype = "complex_")
    for row in range(shape1):
        FT1[row,:] = DFT_one(x[row, :], shape2)
    print('FT1', np.shape(FT1))
    for col in range(shape2):
        FT2[:,col] = DFT_one(FT1[:, col], shape1)

    # FT2 = np.asarray(FT2)
    print('FT2', np.shape(FT2))
    FT = np.array(FT2,dtype = "complex_")
    print(FT.shape)
    return FT / (shape1 * shape2)

# img = cv.imread('./head_phantom_4.png')
img = cv.imread('./coin.png')
shape1 = img.shape[0] # M
shape2 = img.shape[1] # N
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
f = np.fft.fft2(img)
magnitude_spectrum2 = 20*np.log(np.abs(f))
print(np.shape(img))
FT = DFT_two(img)
IM = iDFT_two(FT).real
fig = plt.figure()
ax = Axes3D(fig)
ax.set_zlim3d(0,20)
x= np.arange(0,shape1,1)
y= np.arange(0,shape2,1)
x,y = np.meshgrid(x,y)
# print(x,y)
z = abs(FT)[x,y]
# print(z.shape)
# ax.plot_surface(x,y,z)
#
# plt.show()


# IM = np.fft.ifftn(FT).real

IM2 = np.fft.ifftn(f).real

# FT = np.fft.fftshift(FT)
print(FT.shape)
magnitude_spectrum = 20*np.log(np.abs(FT))
plt.subplot(141),plt.imshow(magnitude_spectrum2, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(IM, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(IM2, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()




