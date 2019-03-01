import cv2
import numpy as np


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

    return X, Y, Z


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
    X = (x - a) * (B - A) / (b - a) + A
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
        Y = l / 903.3 * Y_W

    if v_prime == 0:
        X = 0
        Z = 0
    else:
        X = Y * 2.25 * (u_prime / v_prime)
        Z = Y * (3 - 0.75 * u_prime - 5 * v_prime) / v_prime

    return X, Y, Z


def XYZ_to_sRGB(x, y, z):

    RGB_matrix = np.array(
        [[3.240479, -1.53715, -0.498535], [-0.969256, 1.875991, 0.041556], [0.055648, -2.04043, 1.057311]])
    r, g, b = RGB_matrix.dot(np.array([[x], [y], [z]]))

    def gamma_correction(d):
        if d < 0.00304:
            I = 12.92 * d
        else:
            I = 1.055 * (d ** (1 / 2.4)) - 0.055
        return I

    R = gamma_correction(r) * 255
    G = gamma_correction(g) * 255
    B = gamma_correction(b) * 255

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


array = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                  [[255.0, 0.0, 0.0], [255.0, 0.0, 0.0], [255.0, 0.0, 0.0], [255.0, 0.0, 0.0]],
                  [[100.0, 100.0, 100.0], [100.0, 100.0, 100.0], [100.0, 100.0, 100.0], [100.0, 100.0, 100.0]],
                  [[0.0, 100.0, 100.0], [0.0, 100.0, 100.0], [0.0, 100.0, 100.0], [0.0, 100.0, 100.0]]])

tmp = np.copy(array)
row, column, color = array.shape
L_arrary = np.zeros([row, column, 1])

# program 1

for i in range(row):
    for j in range(column):
        r, g, b = array[i][j]
        x, y, z = sRGB_to_XYZ(r, g, b)
        l, u, v = XYZ_to_Luv(x, y, z)
        L_arrary[i][j] = l
        tmp[i][j] = l, u, v

l_min = np.min(L_arrary)
l_max = np.max(L_arrary)

for i in range(row):
    for j in range(column):
        l, u, v = tmp[i][j]
        L = linear_scaling(l, l_min, l_max, 0, 100)
        X, Y, Z = Luv_to_XYZ(L, u, v)
        R, G, B = XYZ_to_sRGB(X, Y, Z)

# program 2



# program 3

tmp2 = np.copy(array)
Y_arrary = np.zeros([row, column, 1])
for i in range(row):
    for j in range(column):
        r, g, b = array[i][j]
        X, Y, Z = sRGB_to_XYZ(r, g, b)
        x, y, Y = XYZ_to_xyY(X, Y, Z)
        Y_arrary[i][j] = Y
        tmp2[i][j] = x, y, Y

Y_min = np.min(Y_arrary)
Y_max = np.max(Y_arrary)

for i in range(row):
    for j in range(column):
        x, y, Y = tmp2[i][j]
        Y_prime = linear_scaling(Y, Y_min, Y_max, 0, 1)
        X, Y, Z = xyY_to_XYZ(x, y, Y_prime)
        R, G, B = XYZ_to_sRGB(X, Y, Z)







