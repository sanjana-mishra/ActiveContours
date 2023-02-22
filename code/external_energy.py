import cv2
import numpy as np

def line_energy(image):
    #implement line energy (i.e. image intensity)
    E_line = (image - (np.amin(image))) * (1. / (np.amax(image) - np.amin(image)))
    return E_line
    #raise NotImplementedError

def edge_energy(image):
    #implement edge energy (i.e. gradient magnitude)
    gX = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=1, dy=0)
    gY = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=0, dy=1)
    grad_mag = -1 * (np.sqrt((gX**2 + gY**2)))
    #E_edge_norm = cv2.normalize(grad_mag, grad_mag, alpha=1, beta=0, norm_type=cv2.NORM_MINMAX)
    E_edge_norm = (grad_mag - (np.amin(grad_mag))) * (1. / (np.amax(grad_mag) - np.amin(grad_mag)))
    #print(E_edge)
    return E_edge_norm
    #raise NotImplementedError

def term_energy(image):
    #implement term energy (i.e. curvature)
    sX = np.array([[-1, 0, 1]])
    sY = np.array([[-1, 0, 1]]).T

    gX = cv2.filter2D(image, cv2.CV_64F, sX)
    cv2.imshow('gX', gX)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    gY = cv2.filter2D(image, cv2.CV_64F, sY)
    cv2.imshow('gY', gY)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    gXgX = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=2, dy=0)
    cv2.imshow('gXgX', gXgX)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    gYgY = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=0, dy=2)
    cv2.imshow('gYgY', gYgY)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    gXgY = cv2.filter2D(gX, cv2.CV_64F, sY)
    cv2.imshow('gXgY', gXgY)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    numerator = (gXgX * (gY**2)) - (2 * gXgY * gX * gY) + (gYgY * (gX**2))
    denominator = (gX**2 + gY**2) ** (3/2)

    E_term_frac = cv2.divide(numerator, denominator)
    E_term_frac[np.isnan(E_term_frac)] = 0.

    E_term_norm = (E_term_frac - (np.amin(E_term_frac))) * (1./ (np.amax(E_term_frac) - np.amin(E_term_frac)))
    #E_term_norm = cv2.normalize(E_term_frac, None, alpha=1, beta=0, norm_type=cv2.NORM_MINMAX)

    # print(E_term)
    # print(np.amin(E_term))
    # print(np.amax(E_term))

    return E_term_norm

    #raise NotImplementedError

def external_energy(image, w_line, w_edge, w_term):
    #implement external energy
    Energy_term = term_energy(image)
    Energy_line = line_energy(image)
    Energy_edge = edge_energy(image)
    img = image
    cv2.imshow('E_term', Energy_term)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('E_edge', Energy_edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('E_line', Energy_line)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    Energy_external = w_line*Energy_line + w_edge*Energy_edge + w_term*Energy_term
    Energy_external = (Energy_external - (np.amin(Energy_external))) * (1. / (np.amax(Energy_external) - np.amin(Energy_external)))

    cv2.imshow('E_external', Energy_external)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return Energy_external
    #raise NotImplementedError