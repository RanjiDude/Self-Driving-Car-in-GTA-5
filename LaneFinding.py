import numpy as np
from PIL import ImageGrab
import cv2
import time
from matplotlib import pyplot as plt
from directkeys import PressKey, ReleaseKey, W, A, S, D

np.set_printoptions(threshold=np.nan)

def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [0, 0, 255], 10)
    except:
        pass


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, [255,255,255])
    masked = cv2.bitwise_and(img, mask)
    return masked

def perspective_birdseye(original_image):
    pts1 = np.float32([[0, 325], [270, 175], [457, 175], [800, 325]])
    pts2 = np.float32([[0, 600], [0, 0], [800, 0], [800, 600]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(original_image, M, (800, 600))
    return dst

def perspective_normal(original_image):
    pts1 = np.float32([[0, 325], [270, 175], [457, 175], [800, 325]])
    pts2 = np.float32([[0, 600], [0, 0], [800, 0], [800, 600]])
    M = cv2.getPerspectiveTransform(pts2, pts1)
    undst = cv2.warpPerspective(original_image, M, (800, 600))
    return undst

def filter_color(img):
    yellow_min = np.array([65, 80, 80], np.uint8)
    yellow_max = np.array([110, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(img, yellow_min, yellow_max)
    white_min = np.array([0, 0, 200], np.uint8)
    white_max = np.array([255, 80, 255], np.uint8)
    white_mask = cv2.inRange(img, white_min, white_max)
    img = cv2.bitwise_and(img, img, mask=cv2.bitwise_or(yellow_mask, white_mask))
    return img

def window_left(img):
    z = []
    for i in range(400):
        for j in range(600):
            if (img[j][i]) > 100:
                z.append([i, j])
    z = np.asarray(z)
    z = z.reshape((-1,1,2))
    print(z.shape)
    return  z

def window_right(img):
    z = []
    for i in range(401,800):
        for j in range(600):
            if (img[j][i]) > 100:
                z.append([i, j])
    z = np.asarray(z)
    z = z.reshape((-1,1,2))
    print(z.shape)
    return  z

def process_image(original_image):
    vertices = np.array([[10,400],[10,340], [268, 186], [440, 186], [790, 340],[790,400]])

    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    denoised = cv2.GaussianBlur(original_image, (7, 7), 0)

    hls = cv2.cvtColor(denoised, cv2.COLOR_RGB2HLS)

    filtered = filter_color(hls)

    roi_img = roi(filtered, [vertices])

    # sobelx64f = cv2.Sobel(roi_img, cv2.CV_64F, 1, 0, ksize=3)
    # abs_sobel64f = np.absolute(sobelx64f)
    # sobel_8u = np.uint8(abs_sobel64f)

    edges = cv2.Canny(roi_img, 100, 200)

    z_left = window_left(edges)
    z_right = window_right(edges)

    # file = open('z.txt','w')
    # file.write(str(z_left))
    # file.close()

    cv2.polylines(original_image, z_left, True, (0,0,255),3)
    cv2.polylines(original_image, z_right, True, (0, 0, 255), 3)

    # undistort = perspective_normal(distort)

    # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 180, np.array([]), 0, 100)
    # draw_lines(original_image, lines)

    return original_image

tic = time.time()

while(True):
    screen = np.array(ImageGrab.grab(bbox=(0, 30, 810, 630)))
    new_screen = process_image(screen)
    # print('down')
    # PressKey(W)
    # time.sleep(3)
    # print('up')
    # ReleaseKey(W)

    toc = time.time()
    print('Loop took {} seconds'.format(toc - tic))
    tic = time.time()
    # plt.imshow(new_screen, cmap='gray', interpolation='bicubic')
    # plt.show()
    cv2.imshow('window', new_screen)
    # cv2.imshow('window2', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
