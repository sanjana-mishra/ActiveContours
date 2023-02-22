import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

from external_energy import external_energy
from internal_energy_matrix import get_matrix

from scipy import interpolate

def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        #save point
        xs.append(x)
        ys.append(y)

        #display point
        cv2.circle(img, (x, y), 3, 128, -1)
        cv2.imshow('image', img)


if __name__ == '__main__':
    #point initialization
    img_path = '../images/brain.png'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_copy = img.copy()

    xs = []
    ys = []
    cv2.imshow('image', img)

    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #selected points are in xs and ys

    #interpolate
    #implement part 1: interpolate between the  selected points
    n = 200
    num_user_generated_pts = len(xs)
    updated_xs = []
    updated_ys = []
    #current_pt
    #next_pt

    xs = np.r_[xs, xs[0]]
    ys = np.r_[ys, ys[0]]
    contour = np.zeros(((num_user_generated_pts+1), 2))
    contour[:, 0] = xs[:]
    contour[:, 1] = ys[:]
    tck, u = interpolate.splprep(contour.T, s=0, per=1, k=1)
    u_new = np.linspace(u.min(), u.max(), n)
    updated_xs, updated_ys = interpolate.splev(u_new, tck, der=0)

    contours = np.zeros((len(updated_xs), 2))  # 2 for x and y values
    contours[:, 0] = updated_xs
    contours[:, 1] = updated_ys
    contours = contours.reshape((-1, 1, 2)).astype(np.int32)


    # new_generated_pts = np.linspace(current_pt, next_pt, num=n, endpoint=False)
    # draw (display generated pts)



    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img, contours, -1, (255, 0, 0), 3)



    #output window to view newly generated pts
    cv2.imshow('image with points', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img = img_copy.copy()

    smoothed_img = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0)
    #raise NotImplementedError

    alpha = 0.1  # tension
    beta = 0.5  # smoothness
    gamma = 0.5  # step-size
    kappa = 0.5  # external factor
    num_points = len(xs)

    # get matrix
    M = get_matrix(alpha, beta, gamma, n)

    # get external energy
    w_line = 0.5
    w_edge = 1.5
    w_term = 0.5

    E = external_energy(smoothed_img, w_line, w_edge, w_term)


    iterations = 15000
    for i in range(iterations):
        #optimization loop

        ##looping to check and keep points within image shape only
        for j in range(len(updated_xs)):
            if updated_xs[j] < 0:
                updated_xs[j] = 0
            if updated_ys[j] < 0:
                updated_ys[j] = 0
            if updated_ys[j] > img.shape[1]-2:
                updated_ys[j] = img.shape[1]-2
            if updated_xs[j] > img.shape[0]-2:
                updated_xs[j] = img.shape[0]-2



        #gradience along x and y
        E_fx = cv2.Sobel(E, ddepth=cv2.CV_64F, dx=1, dy=0)
        E_fy = cv2.Sobel(E, ddepth=cv2.CV_64F, dx=0, dy=1)
        # print(E_fx.shape)
        # b_fx[:, 0] = E_fx.round()
        # b_fy[:, 1] = E_fy.round()

        # ##Bilinear interpolation
        # b_fx = np.zeros(n)
        # b_fy = np.zeros(n)
        # for i in range(n-1):
        #     x_dec = updated_xs[i] - math.floor(updated_xs[i])
        #     x_prev = math.floor(updated_xs[i])
        #     x_next = math.floor(updated_xs[i]) + 1
        #     y_dec = updated_ys[i] - math.floor(updated_ys[i])
        #     y_prev = math.floor(updated_ys[i])
        #     y_next = math.floor(updated_ys[i]) + 1
        #
        #     print(E_fx.shape)
        #     print(x_prev)
        #     print(y_prev)
        #
        #     x1 = ((1.0 - x_dec) * E_fx[(y_prev, x_prev)]) + (x_dec * E_fx[(x_next, x_prev)])
        #     x2 = ((1.0 - x_dec) * E_fx[(y_prev, x_next)]) + (x_dec * E_fx[(x_next, x_next)])
        #     b_fx[i] = (((1.0 - x_dec) * x1) + (x_dec * x2))

        #     y1 = ((1.0 - x_dec) * E_fy[(y_prev, x_prev)]) + (x_dec * E_fy[(y_next, x_prev)])
        #     y2 = ((1.0 - x_dec) * E_fy[(y_prev, x_next)]) + (x_dec * E_fy[(y_next, x_next)])
        #     b_fy[i] = (((1.0 - y_dec) * y1) + (y_dec * y2))

        x_t = np.dot(M, ((gamma * updated_xs) - (kappa * E_fx[updated_xs.round().astype(np.int32), updated_ys.round().astype(np.int32)])))
        y_t = np.dot(M, ((gamma * updated_ys) - (kappa * E_fy[updated_xs.round().astype(np.int32), updated_ys.round().astype(np.int32)])))
        updated_xs = x_t.copy()
        updated_ys = y_t.copy()

    img = img_copy.copy()
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    contours = np.zeros((len(updated_xs), 2))  # 2 for x and y values
    contours[:, 0] = updated_xs
    contours[:, 1] = updated_ys
    contours = contours.reshape((-1, 1, 2)).astype(np.int32)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    cv2.imshow('Active Contours', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #interpolate.interp2d(updated_xs, updated_ys, contours, kind='linear', copy=True, bounds_error=False, fill_value=None)


    #raise NotImplementedError
