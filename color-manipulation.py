import cv2
import numpy as np
# import sys

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
w2 = 0.05
h2 = 0.05
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

rows, cols, bands = inputImage.shape # bands == 3
W1 = round(w1*(cols-1))
H1 = round(h1*(rows-1))
W2 = round(w2*(cols-1))
H2 = round(h2*(rows-1))

# The transformation should be based on the
# historgram of the pixels in the W1,W2,H1,H2 range.
# The following code goes over these pixels

tmp = np.copy(inputImage)

XYZ_matrix = np.array([[0.412453, 0.35758, 0.180423], [0.212671, 0.71516, 0.072169], [0.019334, 0.119193, 0.950227]])

X_W = 0.95
Y_W = 1.0
Z_W = 1.09
u_w = (4*X_W)/(X_W+15*Y_W+3*Z_W)
v_w = (9*Y_W)/(X_W+15*Y_W+3*Z_W)


for i in range(H1, H2+1) :
    for j in range(W1, W2+1) :
        b, g, r = inputImage[i, j]
        # print(i,j)
        # print(b,g,r)
        # gray = round(0.3*r + 0.6*g + 0.1*b + 0.5)
        # tmp[i, j] = [gray, gray, gray]

        # convert RGB to XYZ
        X, Y, Z = XYZ_matrix.dot(np.array([[r], [g], [b]]))

        # convert XYZ to Luv
        t = Y/Y_W
        if t > 0.008856:
            L = 116*(t**(1/3))-16
        else:
            L = 903.3*t

        d = X + 15*Y + 3*Z
        u_prime = 4*X/d
        v_prime = 9*Y/d

        u = 13*L*(u_prime-u_w)
        v = 13*L*(v_prime - v_w)

        print(L,u,v)


# cv2.imshow('tmp', tmp)

# end of example of going over window

# outputImage = np.zeros([rows, cols, bands], dtype=np.uint8)
#
# for i in range(0, rows) :
#     for j in range(0, cols) :
#         b, g, r = inputImage[i, j]
#         outputImage[i,j] = [b, g, r]
# cv2.imshow("output:", outputImage)
# cv2.imwrite(name_output, outputImage);


# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
