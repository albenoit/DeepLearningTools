import cv2
import numpy as np


def selectpixel(image, invert_xy=False):
    '''
    Create an interactive display of the input image, where the user can select a list of pixels, and get their coordinates
    params:
        image: The input image
        invert_xy: whether to return each pixel coordinates as (x, y) or (y, x)
    '''
    # :mouse callback function
    def click_event(event, x, y, flags, params):
        font = cv2.FONT_HERSHEY_SIMPLEX
        if event == cv2.EVENT_RBUTTONDOWN:
            print(x, ' ', y)
            cv2.putText(img, str(x) + ',' + str(y), (x, y), font, 1, 255, 2)
            cv2.imshow('Input image', img)
            if invert_xy:
                pixels_list.append((y, x))
            else:
                pixels_list.append((x, y))
    pixels_list = []
    print('input image.shape for RoI selection', image.shape)
    if image.shape[:2] != (512, 512):
        print('Image size != (512, 512)')
        img = np.zeros(image.shape)
        img[256:image.shape[0]-256, 256:image.shape[1]-256] = image[256:image.shape[0]-256, 256:image.shape[1]-256]
    else:
        img = image
    print('Considered image.shape for RoI selection', img.shape)
    print('img min, max, dtype', img.min(), img.max(), img.dtype)
    
    cv2.namedWindow('Input image', cv2.WINDOW_NORMAL)
    cv2.imshow('Input image', img)
    cv2.setMouseCallback('Input image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return pixels_list


def selectROI(img, invert_xy=False):
    '''
    Create an interactive display of the input image, where the user can determine a polygon, and get the coordinates of each point
    params:
        image: The input image
        invert_xy: whether to return each pixel coordinates as (x, y) or (y, x)
    '''

    # :mouse callback function
    def draw_roi(event, x, y, flags, param):

        img2 = img.copy()

        if event == cv2.EVENT_LBUTTONDOWN:  # Left click, select point
            if invert_xy:
                pts.append((y, x))
            else:
                pts.append((x, y))

        if event == cv2.EVENT_RBUTTONDOWN:  # Right click to cancel the last selected point
            pts.pop()

        if event == cv2.EVENT_MBUTTONDOWN:  # Display the selected ROI
            mask = np.zeros(img.shape, np.uint8)
            points = np.array(pts, np.int32)
            points = points.reshape((-1, 1, 2))
            #
            mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
            mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))  # for ROI
            mask3 = cv2.fillPoly(mask.copy(), [points], (0.0, 0.0, 1.0))  # for displaying images on the desktop

            show_image = cv2.addWeighted(src1=img, alpha=0.8, src2=mask3.astype(np.float32), beta=0.2, gamma=0)

            cv2.imshow("mask", mask2)
            cv2.imshow("show_img", show_image)
            cv2.waitKey(0)

        if len(pts) > 0:
            # Draw the last point in pts
            cv2.circle(img2, pts[-1], 3, (0, 0, 255), -1)

        if len(pts) > 1:
            #
            for i in range(len(pts) - 1):
                cv2.circle(img2, pts[i], 5, (0, 0, 255), -1)  # x ,y is the coordinates of the mouse click place
                cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)

        cv2.imshow('image', img2)

    pts = []  # for storing points

    img = cv2.cvtColor(np.float32(img), cv2.COLOR_GRAY2BGR)
    cv2.namedWindow('Right-click on pixels of interest, then press escape')
    cv2.imshow('img', img)
    cv2.setMouseCallback('image', draw_roi)
    print("[INFO] Click the left button: select the point, right click: delete the last selected point, "
          "click the middle button: determine the ROI area")
    print("[INFO] Press ESC to quit")
    cv2.destroyAllWindows()

    return pts
