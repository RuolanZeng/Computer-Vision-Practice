import cv2
import numpy as np
import sys
import math


def sRGB_to_XYZ(r, g, b):
    def inverse_gamma(c):
        c_prime = c / 255
        if c_prime < 0.03928:
            return (c_prime / 12.92)
        else:
            return ((c_prime + 0.055) / 1.055) ** 2.4

    R = inverse_gamma(r)
    G = inverse_gamma(g)
    B = inverse_gamma(b)

    XYZ_matrix = np.array(
        [[0.412453, 0.35758, 0.180423], [0.212671, 0.71516, 0.072169], [0.019334, 0.119193, 0.950227]])

    X, Y, Z = XYZ_matrix.dot(np.array([[R], [G], [B]]))

    return float(X), float(Y), float(Z)


def XYZ_to_Luv(x, y, z):
    X_W = 0.95
    Y_W = 1.0
    Z_W = 1.09
    u_w = (4 * X_W) / (X_W + 15 * Y_W + 3 * Z_W)
    v_w = (9 * Y_W) / (X_W + 15 * Y_W + 3 * Z_W)

    t = y / Y_W
    if t > 0.008856:
        L = 116 * (t ** (1 / 3)) - 16
    else:
        L = 903.3 * t

    d = x + 15 * y + 3 * z
    if d == 0:
        u_prime = 0.0
        v_prime = 0.0
    else:
        u_prime = 4 * x / d
        v_prime = 9 * y / d

    u = 13 * L * (u_prime - u_w)
    v = 13 * L * (v_prime - v_w)

    return L, u, v


def linear_scaling(x, a, b, A, B):
    X = ((x - a) * (B - A) / (b - a))+ A
    return X


def Luv_to_XYZ(l, u, v):
    X_W = 0.95
    Y_W = 1.0
    Z_W = 1.09
    u_w = (4 * X_W) / (X_W + 15 * Y_W + 3 * Z_W)
    v_w = (9 * Y_W) / (X_W + 15 * Y_W + 3 * Z_W)

    if l == 0:
        u_prime = 0
        v_prime = 0
    else:
        u_prime = (u + 13 * u_w * l) / (13 * l)
        v_prime = (v + 13 * v_w * l) / (13 * l)

    if l > 7.9996:
        Y = ((l + 16) / 116) ** 3 * Y_W
    else:
        Y = (l / 903.3) * Y_W

    if v_prime == 0:
        X = 0
        Z = 0
    else:
        X = Y * 2.25 * (u_prime / v_prime)
        Z = (Y * (3 - 0.75 * u_prime - 5 * v_prime)) / v_prime

    return X, Y, Z


def XYZ_to_sRGB(x, y, z):

    RGB_matrix = np.array(
        [[3.240479, -1.53715, -0.498535], [-0.969256, 1.875991, 0.041556], [0.055648, -0.204043, 1.057311]])
    r, g, b = RGB_matrix.dot(np.array([[x], [y], [z]]))
    def gamma_correction(d):
        if d < 0.00304:
            I = 12.92 * d
        else:
            I = 1.055 * (d ** (1 / 2.4)) - 0.055

        if I > 1:
            I =1
        if I < 0:
            I = 0
        return I

    R = gamma_correction(r)*255
    G = gamma_correction(g)*255
    B = gamma_correction(b)*255


    return R, G, B


def XYZ_to_xyY(x,y,z):
    if x+y+z==0:
        x_prime = 0
        y_prime = 0
    else:
        x_prime = x/(x+y+z)
        y_prime = y/(x+y+z)
    Y = y

    return x_prime, y_prime, Y


def xyY_to_XYZ(x, y, Y):
    if y == 0:
        X = 0
        Z = 0
    else:
        X = (x/y)*Y
        Z = ((1-x-y)/y)*Y

    return X, Y, Z


def histogram_equalization(L_array):
    L_array = np.floor(L_array)
    set_bins = np.append(np.unique(L_array), (np.max(L_array) + 1))
    hist, bins = np.histogram(L_array.astype(int).flatten(), bins=set_bins)
    cdf = hist.cumsum()
    h2 = []
    for i in range(len(cdf)):
        if i == 0:
            h2.append(math.floor((cdf[i] / 2) * (256 / cdf.max())))
        else:
            h2.append(math.floor(((cdf[i] + cdf[i - 1]) / 2) * (256 / cdf.max())))
    L2 = np.interp(L_array.astype(int).flatten(), bins[:-1], h2)
    result = L2.reshape(L_array.shape)

    return result


# if(len(sys.argv) != 7) :
#     print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
#     print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
#     print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
#     sys.exit()

# w1 = float(sys.argv[1])
# h1 = float(sys.argv[2])
# w2 = float(sys.argv[3])
# h2 = float(sys.argv[4])
# name_input = sys.argv[5]
# name_output = sys.argv[6]

w1 = 0
h1 = 0
w2 = 0.5
h2 = 1
name_input = "good-test-image-for-proj1.bmp"
name_output = "out.png"

#
# if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
#     print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
#     sys.exit()

inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
# if(inputImage is None) :
#     print(sys.argv[0], ": Failed to read image from: ", name_input)
#     sys.exit()

# cv2.imshow("input image: " + name_input, inputImage)

rows, cols, bands = inputImage.shape  # bands == 3
W1 = round(w1 * (cols - 1))
H1 = round(h1 * (rows - 1))
W2 = round(w2 * (cols - 1))
H2 = round(h2 * (rows - 1))

# The transformation should be based on the
# historgram of the pixels in the W1,W2,H1,H2 range.
# The following code goes over these pixels

# program 1

output1 = np.copy(inputImage)
L_arrary = []
tmp = np.zeros(inputImage.shape, dtype=float)


for i in range(H1, H2 + 1):
    for j in range(W1, W2 + 1):
        b, g, r = inputImage[i, j]
        x, y, z = sRGB_to_XYZ(r, g, b)
        l, u, v = XYZ_to_Luv(x, y, z)
        L_arrary.append(l)
        tmp[i][j] = l, u, v

l_min = min(L_arrary)
l_max = max(L_arrary)


for i in range(H1, H2 + 1):
    for j in range(W1, W2 + 1):
        l, u, v = tmp[i][j]
        L = linear_scaling(l, l_min, l_max, 0, 100)
        X, Y, Z = Luv_to_XYZ(L, u, v)
        R, G, B = XYZ_to_sRGB(X, Y, Z)
        output1[i][j] = B, G, R


# cv2.imshow('output1', output1)
# cv2.imwrite('output1.bmp', output1)

# program 2

output2 = np.copy(inputImage)
# L_arrary = np.zeros([], dtype=float)
tmp2 = np.zeros(inputImage.shape, dtype=float)


for i in range(H1, H2 + 1):
    for j in range(W1, W2 + 1):
        b, g, r = inputImage[i, j]
        x, y, z = sRGB_to_XYZ(r, g, b)
        l, u, v = XYZ_to_Luv(x, y, z)
        # L_arrary.append(l)
        tmp2[i][j] = l, u, v


for i in range(H1, H2 + 1):
    for j in range(W1, W2 + 1):
        l, u, v = tmp2[i][j]
        L = l
        # L = linear_scaling(l, l_min, l_max, 0, 100)
        X, Y, Z = Luv_to_XYZ(L, u, v)
        R, G, B = XYZ_to_sRGB(X, Y, Z)
        output2[i][j] = B, G, R

# cv2.imshow('output1', output1)
cv2.imwrite('output2.bmp', output2)


# program 3

output3 = np.copy(inputImage)
tmp3 = np.zeros(inputImage.shape, dtype=float)
Y_arrary = []

for i in range(H1, H2 + 1):
    for j in range(W1, W2 + 1):
        b, g, r = inputImage[i, j]
        X, Y, Z = sRGB_to_XYZ(r, g, b)
        x, y, Y = XYZ_to_xyY(X, Y, Z)
        Y_arrary.append(Y)
        tmp3[i][j] = x, y, Y

Y_min = min(Y_arrary)
Y_max = max(Y_arrary)

for i in range(H1, H2 + 1):
    for j in range(W1, W2 + 1):
        x, y, Y = tmp3[i][j]
        Y_prime = linear_scaling(Y, Y_min, Y_max, 0, 1)
        X, Y, Z = xyY_to_XYZ(x, y, Y_prime)
        R, G, B = XYZ_to_sRGB(X, Y, Z)
        output3[i][j] = B, G, R

# cv2.imshow('output3', output3)
# cv2.imwrite('output3.bmp', output3)


# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
