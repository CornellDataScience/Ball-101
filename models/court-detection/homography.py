import os
import cv2 as cv
import numpy as np

PATH = os.path.join('models','court-detection')
TEMP_PATH = os.path.join(PATH,'temp')
COURT_PATH = os.path.join(TEMP_PATH,'court.jpeg')
TEMPLATE_PATH = os.path.join(TEMP_PATH,'template.jpeg')

def main():
    # test_homography(COURT_PATH)
    bgr_image = cv.imread(COURT_PATH)
    ycrcb_img = cv.cvtColor(bgr_image,cv.COLOR_BGR2YCrCb)

    bins = bin_pixels(ycrcb_img, cr_bins=10, cb_bins=10)

    mask = get_mask(ycrcb_img, bins[1])
    res = apply_mask(bgr_image,mask)

    contours = get_hull(mask)
    res2 = apply_hull(bgr_image,contours)

    # convert_to_hsv(COURT_PATH)
    # convert_to_ycrcb(COURT_PATH)

    cv.imshow("original",bgr_image)
    cv.imshow("mask",mask)
    cv.imshow("masked image",res)
    cv.imshow("image with contours",res2)

    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

class YCrBrBin:
    def __init__(self, value, cr_lower:int, cr_upper:int, cb_lower:int, cb_upper:int):
        self.cr_lower = cr_lower
        self.cr_upper = cr_upper
        self.cb_lower = cb_lower
        self.cb_upper = cb_upper
        self.value = value

    def __str__(self):
        return 'YCrCbBin('+str(round(self.value,3))+','+str(self.cr_lower)+','+str(self.cr_upper)+','+str(self.cb_lower)+','+str(self.cb_upper)+')'

# Returns top bins from YCrCb color space
# @Precondition: img is YCrCb converted image
# @Returns: sorted array of most frequent YCrCb bin objects
def bin_pixels(img:np.ndarray, cr_bins:int=16, cb_bins:int=16):
    # generate weights
    weights = np.zeros((img.shape[0],img.shape[1]))
    for row in range(weights.shape[0]):
        row_weight = np.full(weights.shape[1],min(row+1,weights.shape[0]-row))
        weights[row] = row_weight
    for col in range(weights.shape[1]):
        col_weight = np.full(weights.shape[0],min(col+1,weights.shape[1]-col))
        weights[:,col] = np.minimum(weights[:,col],col_weight)
    weights = (weights*(weights.shape[0]/2.0))**2
    weights = weights/np.sum(weights) 

    # split image pixels into bins
    bins = np.zeros((cr_bins,cb_bins))
    cr_step = 255.0 / cr_bins
    cb_step = 255.0 / cb_bins
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            pix = img[row,col]
            bins[int(pix[1]/cr_step),int(pix[2]/cb_step)] += weights[row,col]

    # sort bins
    top_bins = []
    THRESHOLD = 0.001
    for row in range(bins.shape[0]):
        for col in range(bins.shape[1]):
            if (bins[row,col] <= THRESHOLD):
                continue
            top_bins.append(YCrBrBin(bins[row,col],
                            int(round(cr_step*row)),
                            int(round(cr_step*(row+1))),
                            int(round(cb_step*col)),
                            int(round(cb_step*(col+1)))))
    return sorted(top_bins, reverse=True, key=lambda bin: bin.value)
    
# applies color range from top bin to image
def get_mask(img:np.ndarray, bin:YCrBrBin):
    # get mask
    lowerbound = np.array([0,bin.cr_lower,bin.cb_lower])
    upperbound = np.array([255,bin.cr_upper,bin.cb_upper])
    mask = cv.inRange(img, lowerbound, upperbound)
    cv.imshow('original mask',mask)

    # close and open mask
    kernel = np.ones((3,3),np.uint8)
    mask = cv.morphologyEx(mask,cv.MORPH_CLOSE,kernel,iterations=4)
    mask = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel,iterations=3)
    return mask 

# @Precondition: img should be BGR color space
def apply_mask(img:np.ndarray, mask:np.ndarray):
    return cv.bitwise_and(img,img,mask=mask)

def get_hull(mask:np.ndarray):
    ret, thresh = cv.threshold(mask,127,255,0)
    contours, hierachy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours

def apply_hull(img:np.ndarray, contours:tuple):
    copy = img.copy()
    cv.drawContours(copy, contours, -1, (0,255,0), 1)
    return copy

def test_homography(source_image:str):
    im_src = cv.imread(source_image)
    pts_src = np.array([[198,169],[482,236],[418,322],[87,221]])

    im_dst = cv.imread(TEMPLATE_PATH)
    pts_dst = np.array([[38,13],[571,13],[571,339],[38,339]])

    h, _ = cv.findHomography(pts_src,pts_dst)
    im_out = cv.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
    
    cv.imshow("Source Image", im_src)
    cv.imshow("Destination Image", im_dst)
    cv.imshow("Warped Source Image", im_out)  

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
    luma_only[:,:,1] = 128
    luma_only[:,:,2] = 128

    blue_only = ycrcb.copy()
    blue_only[:,:,0] = 0
    blue_only[:,:,2] = 128

    red_only = ycrcb.copy()
    red_only[:,:,0] = 0
    red_only[:,:,1] = 128

    luma_only = cv.cvtColor(luma_only, cv.COLOR_YCrCb2BGR)
    blue_only = cv.cvtColor(blue_only, cv.COLOR_YCrCb2BGR)
    red_only = cv.cvtColor(red_only, cv.COLOR_YCrCb2BGR)

    cv.imshow("Original", img)
    cv.imshow("Luma Only", luma_only)
    cv.imshow("Blue Difference Only", blue_only)
    cv.imshow("Red Difference Only", red_only)


if __name__ == '__main__':
    main()