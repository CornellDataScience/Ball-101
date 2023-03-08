import os
import cv2 as cv
import numpy as np

PATH = os.path.join('models','court-detection')
TEMP_PATH = os.path.join(PATH,'temp')
COURT_PATH = os.path.join(TEMP_PATH,'court.jpeg')
TEMPLATE_PATH = os.path.join(TEMP_PATH,'template.jpeg')

def main():
    img = cv.imread(COURT_PATH)
    bins = bin_pixels(img, cr_bins=16, cb_bins=16)
    masked_img = apply_mask(img, bins)
    # test_homography(COURT_PATH)
    # harris_corners(COURT_PATH)
    # tomasi_corners(COURT_PATH)
    # sift_corners(COURT_PATH)

    # harris_corners(TEMPLATE_PATH)
    # sift_corners(TEMPLATE_PATH)

    # convert_to_hsv(COURT_PATH)
    # convert_to_ycrcb(COURT_PATH)

    cv.imshow("original", img)
    cv.imshow("masked image", masked_img)

    if cv.waitKey(0) & 0xff == 27:
        print('got here')
        cv.destroyAllWindows()

# Returns top bins from YCrCb color space
def bin_pixels(img:np.ndarray, cr_bins:int=16, cb_bins:int=16):
    # prepare bin
    bins = [[[0,0,0] for j in range(cb_bins)] for i in range(cr_bins)]
    cr_step = 255.0 / cr_bins
    cb_step = 255.0 / cb_bins
    cr_lower, cr_upper = 0, cr_step
    for row in range(cr_bins):
        cb_lower, cb_upper = 0, cb_step
        for col in range(cb_bins):
            bins[row][col][1] = (int(round(cr_lower)),int(round(cr_upper)))
            bins[row][col][2] = (int(round(cb_lower)),int(round(cb_upper)))
            cb_lower += cb_step
            cb_upper += cb_step
        cb_lower
        cr_lower += cr_step
        cr_upper += cr_step

    # split image pixels into bins
    src = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    for line in src:
        for pix in line: # TODO: implement scaling around part of video
            bins[int(pix[1]/cr_step)][int(pix[2]/cb_step)][0] += 1

    # normalize counts
    total = 0
    for row in bins:
        for bin in row:
            total += bin[0]
    for row in bins:
        for bin in row:
            bin[0] = round(bin[0] / float(total), ndigits=4)

    # sort bins
    top_bins = [bin for row in bins for bin in row if bin[0]>0.001]
    sorted_bins = sorted(top_bins, reverse=True)
    return sorted_bins
    
# applies color range from top bin to image
def apply_mask(img:np.ndarray, bins:list):
    # get mask
    lower_cr, upper_cr = bins[0][1]
    lower_cb, upper_cb = bins[0][2]
    lowerbound = np.array([0,lower_cr,lower_cb])
    upperbound = np.array([255,upper_cr,upper_cb])
    src = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    mask = cv.inRange(src, lowerbound, upperbound)
    cv.imshow("masked image", mask)

    # apply mask
    res = cv.bitwise_and(src,src,mask=mask)
    return cv.cvtColor(res,cv.COLOR_YCrCb2BGR)




def test_homography(source_image:str):
    im_src = cv.imread(source_image)
    pts_src = np.array([[198,169],[482,236],[418,322],[87,221]])

    im_dst = cv.imread(TEMPLATE_PATH)
    pts_dst = np.array([[38,13],[571,13],[571,339],[38,339]])

    h, status = cv.findHomography(pts_src,pts_dst)
    im_out = cv.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
    
    cv.imshow("Source Image", im_src)
    cv.imshow("Destination Image", im_dst)
    cv.imshow("Warped Source Image", im_out)  

def harris_corners(source_image:str):
    img = cv.imread(source_image)

    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray_arr = np.float32(gray)
    dst = cv.cornerHarris(gray_arr,20,3,0.04)

    # dilate to original size
    dst = cv.dilate(dst,None)
    harris = img.copy()
    harris[dst>0.05*dst.max()]=[0,0,255]

    cv.imshow('Harris Corner Detector',harris)

def tomasi_corners(source_image:str):
    img = cv.imread(source_image)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    corners = cv.goodFeaturesToTrack(gray,25,0.01,10)
    corners = np.int0(corners)

    #draw circles onto image
    tomasi = img.copy()
    for i in corners:
        x,y = i.ravel()
        cv.circle(tomasi,(x,y),3,255,-1)

    cv.imshow('Shi-Tomasi Corner Detector',tomasi)

def sift_corners(source_image:str):
    img = cv.imread(source_image)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp = sift.detect(gray,None)

    sift_img = cv.drawKeypoints(gray,kp,img)

    cv.imshow('SIFT Corner Detection',sift_img)

def convert_to_hsv(source_image:str):
    img = cv.imread(source_image)

    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)

    hue_only = hsv.copy()
    hue_only[:,:,1] = 255
    hue_only[:,:,2] = 255

    sat_only = hsv.copy()
    sat_only[:,:,0] = 0
    sat_only[:,:,2] = 255

    val_only = hsv.copy()
    val_only[:,:,0] = 0
    val_only[:,:,1] = 255

    hue_only = cv.cvtColor(hue_only, cv.COLOR_HSV2BGR)
    sat_only = cv.cvtColor(sat_only, cv.COLOR_HSV2BGR)
    val_only = cv.cvtColor(val_only, cv.COLOR_HSV2BGR)

    cv.imshow("Original", img)
    cv.imshow("Hue Only", hue_only)
    cv.imshow("Saturation Only", sat_only)
    cv.imshow("Value Only", val_only)

def convert_to_ycrcb(source_image:str):
    img = cv.imread(source_image)

    ycrcb = cv.cvtColor(img,cv.COLOR_BGR2YCrCb)

    luma_only = ycrcb.copy()
    luma_only[:,:,1] = 125
    luma_only[:,:,2] = 125

    blue_only = ycrcb.copy()
    blue_only[:,:,0] = 125
    blue_only[:,:,2] = 125

    red_only = ycrcb.copy()
    red_only[:,:,0] = 125
    red_only[:,:,1] = 125

    luma_only = cv.cvtColor(luma_only, cv.COLOR_YCrCb2BGR)
    blue_only = cv.cvtColor(blue_only, cv.COLOR_YCrCb2BGR)
    red_only = cv.cvtColor(red_only, cv.COLOR_YCrCb2BGR)

    cv.imshow("Original", img)
    cv.imshow("Luma Only", luma_only)
    cv.imshow("Blue Difference Only", blue_only)
    cv.imshow("Red Difference Only", red_only)


if __name__ == '__main__':
    main()