import os
import cv2 as cv
import numpy as np
    
PATH = os.path.join('models','court-detection')
TEMP_PATH = os.path.join(PATH,'temp')
VIDEO_PATH = os.path.join(TEMP_PATH,'demo.mov')
COURT_PATH = os.path.join(TEMP_PATH,'court.jpeg')
TEMPLATE_PATH = os.path.join(TEMP_PATH,'template.jpeg')


USE_HSV = True
CUSHION_BIN = 5

_, bgr_img = cv.VideoCapture(VIDEO_PATH).read()
ycrcb_img = cv.cvtColor(bgr_img,cv.COLOR_BGR2YCrCb)
hsv_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2HSV)
gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)

if USE_HSV:
    index = (0,1)
    one_max = 180.0
    two_max = 256.0
    img = hsv_img
else:
    index = (1,2)
    one_max = 256.0
    two_max = 256.0
    img = ycrcb_img

def main():
    # test_homography(COURT_PATH)

    # bins = bin_pixels(img, one_bins=16, two_bins=16)
    bins = bin_pixels(img, one_bins=18, two_bins=10)


    # test_many_bins(img, bgr_image, bins, iterations=1)

    mask = get_mask(img, bins[0])
    canny_edges = get_canny(gray_img)
    masked_edges = apply_mask(canny_edges, mask)
    hough_lines = get_hough(masked_edges)
    hough = apply_hough(bgr_img, hough_lines)

    # contours = get_hull(mask)
    # res2 = apply_hull(bgr_image,contours)

    # # convert_to_hsv(COURT_PATH)
    # # convert_to_ycrcb(COURT_PATH)
    # convert_to_ycrcb('none',alt=bgr_image)
    # convert_to_hsv('none',alt=bgr_image)

    # test_many_canny(gray_img, mask, [10,50,200])
    # test_many_hough(bgr_img, masked_edges, [[1],[np.pi/180],[150,200,250]])
    # test_many_hough(bgr_img, masked_edges, [[1],[0.02,0.04,0.06],[200]])
    # test_many_hough(bgr_img, masked_edges, [[0.5,1,2,4],[np.pi/180],[200]])

    cv.imshow('original', bgr_img)
    cv.imshow('mask', mask)
    cv.imshow('canny', canny_edges)
    cv.imshow('canny masked', masked_edges)
    cv.imshow('hough transform', hough)

    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

class Bin:
    def __init__(self, value, one_lower:int, one_upper:int, two_lower:int, two_upper:int):
        self.one_lower = one_lower
        self.one_upper = one_upper
        self.two_lower = two_lower
        self.two_upper = two_upper
        self.value = value

    def __str__(self):
        return 'Bin('+str(round(self.value,3))+','+str(self.one_lower)+','+str(self.one_upper)+','+str(self.two_lower)+','+str(self.two_upper)+')'

# Returns top bins from YCrCb color space
# @Precondition: img is YCrCb converted image
# @Returns: sorted array of most frequent YCrCb bin objects
def bin_pixels(img:np.ndarray, one_bins:int=16, two_bins:int=16):
    # generate weights
    weights = np.zeros((img.shape[0],img.shape[1]))
    for row in range(weights.shape[0]):
        row_weight = np.full(weights.shape[1], row)
        weights[row] = row_weight
    for col in range(weights.shape[1]):
        col_weight = np.full(weights.shape[0], min(col+1,weights.shape[1]-col))
        weights[:,col] = np.minimum(weights[:,col],col_weight)
    weights = weights/np.sum(weights) 

    # split image pixels into bins
    bins = np.zeros((one_bins,two_bins))
    one_step = one_max / one_bins
    two_step = two_max / two_bins
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            pix = img[row,col]
            bins[int(pix[index[0]]/one_step),int(pix[index[1]]/two_step)] += weights[row,col]

    # sort bins
    top_bins = []
    THRESHOLD = 0.001
    for row in range(bins.shape[0]):
        for col in range(bins.shape[1]):
            if (bins[row,col] <= THRESHOLD):
                continue
            top_bins.append(Bin(bins[row,col],
                            int(round(one_step*row)),
                            int(round(one_step*(row+1))),
                            int(round(two_step*col)),
                            int(round(two_step*(col+1)))))
    return sorted(top_bins, reverse=True, key=lambda bin: bin.value)
    
# applies color range from top bin to image
# @Precondition img must be same color space as bin was found in
def get_mask(img:np.ndarray, bin:Bin):
    # get mask
    lowerbound = np.full(3,0)
    upperbound = np.full(3,255)
    lowerbound[index[0]] = bin.one_lower - CUSHION_BIN*1
    lowerbound[index[1]] = bin.two_lower - CUSHION_BIN*1
    upperbound[index[0]] = bin.one_upper + CUSHION_BIN*0
    upperbound[index[1]] = bin.two_upper + CUSHION_BIN*0
    mask = cv.inRange(img, lowerbound, upperbound)
    cv.imshow('original mask',mask)

    # close and open mask
    kernel = np.ones((3,3),np.uint8)
    mask = cv.morphologyEx(mask,cv.MORPH_CLOSE,kernel,iterations=8) #get all court
    mask = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel,iterations=30) #remove distractions
    return mask 

# @Precondition: img should be BGR color space
def apply_mask(img:np.ndarray, mask:np.ndarray):
    return cv.bitwise_and(img,img,mask=mask)


# Returns image after running through canny edge detector
# @Precondition: must be 8-bit gray scale image
def get_canny(img:np.ndarray, threshold1=10, threshold2=100):
    return cv.Canny(img,threshold1,threshold2)

# Returns list of lines to draw through hough transform
# Preconditino: Should be 8-bit grayscale image
def get_hough(img:np.ndarray,rho:float=1,theta:float=np.pi/180,threshold:int=200):
    lines = cv.HoughLines(img, rho, theta, threshold)
    return lines

def apply_hough(img:np.ndarray, lines:list):
    out = img.copy()
    for line in lines:
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a*rho, b*rho
        x1, y1 = int(x0 + 2000*(-b)), int(y0 + 2000*(a))
        x2, y2 = int(x0 - 2000*(-b)), int(y0 - 2000*(a)) 
        cv.line(out,(x1,y1),(x2,y2),[0,0,255])
    return out


def get_hull(img:np.ndarray):
    ret, thresh = cv.threshold(img,127,255,0)
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

def convert_to_hsv(source_image:str, alt:np.ndarray=None):
    if alt is None:
        img = cv.imread(source_image)
    else:
        img = alt

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

def convert_to_ycrcb(source_image:str, alt:np.ndarray=None):
    if alt is None:
        img = cv.imread(source_image)
    else:
        img = alt

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

def test_many_bins(img:np.ndarray, bgr:np.ndarray, bins:list, iterations:int=6):
    for i in range(iterations):
        mask = get_mask(img, bins[i])
        masked = apply_mask(bgr,mask)
        cv.imshow('Bin Level ' + str(i), masked)

def test_many_canny(img:np.ndarray, mask:np.ndarray, grid:list):
    for one in grid:
        for two in grid:
            if one > two:
                continue
            canny = get_canny(img, threshold1=one, threshold2=two)
            masked_canny = apply_mask(canny, mask)
            cv.imshow(str(one)+' by '+str(two), masked_canny) 

def test_many_hough(img:np.ndarray, canny:np.ndarray, grid:list):
    for rho in grid[0]:
        for theta in grid[1]:
            for threshold in grid[2]:
                lines = get_hough(canny, rho, theta, threshold)
                hough = apply_hough(img, lines)
                cv.imshow(str(rho)+' by '+str(round(theta,3))+' by '+str(threshold), hough)

if __name__ == '__main__':
    main()