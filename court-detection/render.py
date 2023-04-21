import os
import cv2 as cv
import numpy as np
import random


class Bin:
    'Bin to store color ranges of two channels'
    def __init__(self, value, one_lower:int, one_upper:int, two_lower:int, two_upper:int):
        self.one_lower = one_lower
        self.one_upper = one_upper
        self.two_lower = two_lower
        self.two_upper = two_upper
        self.value = value

    def __str__(self):
        return 'Bin('+str(round(self.value,3))+','+str(self.one_lower)+','+str(self.one_upper)+','+str(self.two_lower)+','+str(self.two_upper)+')'


class Render:
    def __init__(self, video_path:str, display_images:bool=False):
        'Initializes all paths to images and truth values and runs court detection'
        self.TRUE_PATH = os.path.join('court-detection','temp','true_map.png')
        self.VIDEO_PATH = video_path
        self.BINNING_THRESHOLD = 0.001
        self.COLOR_SMOOTHING = 5
        self.HALF_COURT_BOUNDARY = np.array([(2043,1920),(2043,35),(37,35),(37,1920)])
        self.BOX_BOUNDARY = np.array([(1277,798),(1277,35),(803,35),(803,798)])
        video = cv.VideoCapture(self.VIDEO_PATH)
        _, self.bgr_img = video.read()
        self.ycrcb_img = cv.cvtColor(self.bgr_img,cv.COLOR_BGR2YCrCb)
        self.hsv_img = cv.cvtColor(self.bgr_img, cv.COLOR_BGR2HSV)
        self.gray_img = cv.cvtColor(self.bgr_img, cv.COLOR_BGR2GRAY)
        self.masked_edges = self.gray_img.copy()
        self.true_map = cv.imread(self.TRUE_PATH,cv.IMREAD_GRAYSCALE)

        self.HSV_BINNING = True # choose either HSV binning or YCrCb binning
        if self.HSV_BINNING:
            self.index = (0,1)
            self.one_max = 180.0
            self.two_max = 256.0
            self.img = self.hsv_img
        else:
            self.index = (1,2)
            self.one_max = 256.0
            self.two_max = 256.0
            self.img = self.ycrcb_img

        self.DISPLAY_IMAGES = display_images
        if self.DISPLAY_IMAGES == True:
            self.homography = self.detect_courtlines_and_display()
        else:
            self.homograpy = self.detect_courtlines()

    def render_video(self,states:list[dict],players:list[str],filename:str):
        '''
        Takes into player position data, applied homography, and renders video stored in filename
        @Preconditions, states is list of dictionary each with key "frame_no"
        '''
        # Create a blank image to use as the background for each frame
        background = cv.cvtColor(self.true_map,cv.COLOR_GRAY2BGR)
        height, width, _ = background.shape
        
        # Initialize the video writer
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_writer = cv.VideoWriter(filename, fourcc, 30, (width,height))

        # Define initial positions for each player
        player_state = {}
        for player in players:
            player_state.update({player:{'pos':(0,0),
                                        'color':(random.randint(0,256),random.randint(0,256),random.randint(0,256))}})
        
        # find duration of video
        dur = 0
        for frame in states:
            if frame['frameno'] > dur:
                dur = frame['frameno']

        frame_index = 0

        # Loop through each time step
        for t in range(1,dur+30):
            # Create a copy of the background image to draw the points on
            frame = background.copy()

            # Get dictionary of positions at each frame 
            if states[frame_index] == t:
                state = states[t]
                for player in players:
                    if player in state:
                        pd = state[player]
                        x, y = (pd['xmin']+pd['xmax'])/2.0, pd['ymax']
                        x, y = self.transform_point(x,y)
                        player_state[player].update({'pos':(int(x), int(y))})
                frame_index += 1
            
            # Loop through each point and draw it on the frame
            for player in players:
                pos = player_state[player]['pos']
                color = player_state[player]['color']
                font = cv.FONT_HERSHEY_SIMPLEX
                thickness = 2
                font_scale = 1
                radius = 20
                text_width = cv.getTextSize(player, font, font_scale, thickness)[0][0]
                cv.circle(img=frame, center=pos, radius=radius, color=color, thickness=-1)
                cv.putText(img=frame,text=player,org=(pos[0]-(text_width//2),pos[1]-radius-10),
                           fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=1,color=color,thickness=2,lineType=cv.LINE_AA)

            # Write the frame to the video writer
            video_writer.write(frame)

        # Release the video writer
        video_writer.release()

    def detect_courtlines(self):
        'Finds best homography and store it as self.homography'
        bins = self.bin_pixels(self.img, one_bins=18, two_bins=10)
        mask = self.get_mask(self.img, bins[0])
        canny_edges = self.get_canny(self.gray_img)
        masked_edges = self.apply_mask(canny_edges, mask)
        hough_lines = self.get_hough(masked_edges,threshold=180)
        thick_masked_edges = self.thicken_edges(masked_edges,iterations=1)
        self.masked_edges = thick_masked_edges.copy()
        best_pts = self.find_best_homography(hough_lines)
        while True:
            if not self.regress_box_boundary(best_pts):
                break
        while True:
            if not self.fine_regress_box_boundary(best_pts):
                break
        homography, _ = cv.findHomography(np.array(best_pts),self.BOX_BOUNDARY)
        return homography

    def detect_courtlines_and_display(self):
        'Finds best homography and displays images of progress'
        bins = self.bin_pixels(self.img, one_bins=18, two_bins=10)
        mask = self.get_mask(self.img, bins[0])
        canny_edges = self.get_canny(self.gray_img)
        masked_edges = self.apply_mask(canny_edges, mask)
        hough_lines = self.get_hough(masked_edges,threshold=180)
        hough = self.apply_hough(self.bgr_img, hough_lines)
        thick_masked_edges = self.thicken_edges(masked_edges,iterations=1)
        self.masked_edges = thick_masked_edges.copy()
        best_pts = self.find_best_homography(hough_lines)
        while self.regress_box_boundary(best_pts):
            print('new goodness',self.evaluate_homography(best_pts,self.BOX_BOUNDARY))
        print('to fine tuning')
        while self.fine_regress_box_boundary(best_pts):
            print('new goodness',self.evaluate_homography(best_pts,self.BOX_BOUNDARY))
        homography, _ = cv.findHomography(np.array(best_pts),self.BOX_BOUNDARY)

        test_bgr = self.bgr_img.copy()
        color_map = cv.cvtColor(self.true_map,cv.COLOR_GRAY2BGR)
        colors = [(0,0,255),(0,255,0),(255,0,0), (255,255,0)]
        for i in range(0,4):
            pt = best_pts[i]
            cv.circle(test_bgr,(int(pt[0]),int(pt[1])), 5, colors[i], -1)
            pt = self.BOX_BOUNDARY[i]
            cv.circle(color_map,(int(pt[0]),int(pt[1])), 10, colors[i], -1)
        new_img = self.apply_bgr_homography(self.bgr_img,best_pts)
        new_gray_img = self.apply_gray_homography(self.masked_edges,best_pts,or_mask=True)
        second_gray_img = self.apply_gray_homography(self.masked_edges,best_pts,or_mask=False)

        
        cv.imshow('original', self.bgr_img)
        cv.imshow('mask', mask)
        cv.imshow('canny', canny_edges)
        cv.imshow('canny masked', masked_edges)
        cv.imshow('hough transform', hough)
        cv.imshow('new test', new_img)
        cv.imshow('gray union', new_gray_img)
        cv.imshow('gray intersection', second_gray_img)
        cv.imshow('points image', test_bgr)
        cv.imshow('true map',color_map)

        if cv.waitKey(0) & 0xff == 27:
            cv.destroyAllWindows()

        return homography

    def bin_pixels(self,img:np.ndarray, one_bins:int=16, two_bins:int=16):
        '''
        Returns top bins from YCrCb color space
        @Precondition: img is YCrCb converted image
        @Returns: sorted array of most frequent YCrCb bin objects
        '''
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
        one_step = self.one_max / one_bins
        two_step = self.two_max / two_bins
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                pix = img[row,col]
                bins[int(pix[self.index[0]]/one_step),
                     int(pix[self.index[1]]/two_step)] += weights[row,col]

        # sort bins
        top_bins = []
        for row in range(bins.shape[0]):
            for col in range(bins.shape[1]):
                if (bins[row,col] <= self.BINNING_THRESHOLD):
                    continue
                top_bins.append(Bin(bins[row,col],
                                int(round(one_step*row)),
                                int(round(one_step*(row+1))),
                                int(round(two_step*col)),
                                int(round(two_step*(col+1)))))
        return sorted(top_bins, reverse=True, key=lambda bin: bin.value)
        
    def get_mask(self,img:np.ndarray, bin:Bin):
        '''
        Applies color range from specified bin
        @Precondition img must be same color space as bin was found in
        '''
        # get mask
        lowerbound = np.full(3,0)
        upperbound = np.full(3,255)
        lowerbound[self.index[0]] = bin.one_lower - self.COLOR_SMOOTHING*1
        lowerbound[self.index[1]] = bin.two_lower - self.COLOR_SMOOTHING*1
        upperbound[self.index[0]] = bin.one_upper + self.COLOR_SMOOTHING*0
        upperbound[self.index[1]] = bin.two_upper + self.COLOR_SMOOTHING*0
        mask = cv.inRange(img, lowerbound, upperbound)

        # close and open mask
        kernel = np.ones((3,3),np.uint8)
        mask = cv.morphologyEx(mask,cv.MORPH_CLOSE,kernel,iterations=8) #get all court
        return cv.morphologyEx(mask,cv.MORPH_OPEN,kernel,iterations=30) #remove distractions

    def apply_mask(self,img:np.ndarray, mask:np.ndarray):
        'Applies bitwise and mask'
        return cv.bitwise_and(img,img,mask=mask)

    def get_canny(self,img:np.ndarray, threshold1=10, threshold2=100):
        'Returns canny edge detected image, two thresholds for two passes'
        return cv.Canny(img,threshold1,threshold2)

    def thicken_edges(self,img:np.ndarray, iterations:int=2):
        '''
        Returns dilated image, means for truth image map
        @Precondition: input should be gray image mask
        '''
        kernel = np.ones((3,3),np.uint8)
        return cv.morphologyEx(img,cv.MORPH_DILATE,kernel,iterations=iterations)

    def get_hough(self,img:np.ndarray,rho:float=1,theta:float=np.pi/180,threshold:int=200):
        '''
        Returns list of lines to draw through hough transform
        @Precondition: Should be 8-bit grayscale image
        '''
        return cv.HoughLines(img, rho, theta, threshold)

    def find_best_homography(self,lines:list):
        '''
        Return four points of inner box with best homography intersection
        @Precondition: lines is list of lines from Hough Transform and img is BGR image of court'''
        # divide into two classes of lines, sorted by rho
        hor = lines[lines[:,0,1]<=np.pi/2] # baseline
        ver = lines[lines[:,0,1]>np.pi/2] # sideline
        hor = np.array(sorted(hor, key = lambda x : x[0][0]))
        ver = np.array(sorted(ver, key = lambda x : x[0][0]))

        print('size before,',len(hor),len(ver))

        # remove similar lines 
        RHO_THRESHOLD = 10
        THETA_THRESHOLD = 2.001 * (np.pi/180)
        # hor = self.filter_similar_lines(hor,rho_threshold=RHO_THRESHOLD,theta_threshold=THETA_THRESHOLD)
        # ver = self.filter_similar_lines(ver,rho_threshold=RHO_THRESHOLD,theta_threshold=THETA_THRESHOLD)

        print('size after,',len(hor),len(ver))
        # print (np.array(ver),np.array(hor))

        max_goodness = 0
        max_homography = None
        for i1 in range(0,len(hor)):
            for i2 in range(i1+1,len(hor)):
                for j1 in range(len(ver)):
                    for j2 in range(j1+1,len(ver)):
                        pts = self.get_four_intersections(hor[i2][0],ver[j1][0],hor[i1][0],ver[j2][0])
                        if pts is None:
                            continue
                        goodness = self.evaluate_homography(pts,self.BOX_BOUNDARY)
                        if (goodness > max_goodness):
                            max_goodness = goodness
                            max_homography = pts

                        pts = self.get_four_intersections(ver[j2][0],hor[i2][0],ver[j1][0],hor[i1][0])
                        if pts is None:
                            continue
                        goodness = self.evaluate_homography(pts,self.BOX_BOUNDARY)
                        if (goodness > max_goodness):
                            max_goodness = goodness
                            max_homography = pts
        print('max goodness: ',max_goodness)
        return max_homography
    
    def evaluate_homography(self,pts_src,pts_dst):
        mapped_edge_img = self.apply_gray_homography(self.masked_edges,pts_src,pts_dst=pts_dst)
        total = 171053
        # total = np.count_nonzero(self.invert_grayscale(self.true_map)
        goodness = float(np.count_nonzero(mapped_edge_img > 100)) / total
        return goodness
            
    def get_four_intersections(self,l1,l2,l3,l4):
        '''
        return intersection of four lines
        OR returns None is intersection of four lines it too close to each other
        '''
        p1 = self.get_line_intersection(l1,l2)
        p2 = self.get_line_intersection(l2,l3)
        p3 = self.get_line_intersection(l3,l4)
        p4 = self.get_line_intersection(l4,l1)
        d1 = self.distance(p1,p2)
        d2 = self.distance(p2,p3)
        d3 = self.distance(p3,p4)
        d4 = self.distance(p4,p1)
        if (d1<600 or d1>800 or d3<600 or d3>800 or d2<50 or 
            d2>300 or d4<50 or d4>300 or self.is_not_convex(p1,p2,p3,p4)):
            return None
        return (p1,p2,p3,p4)

    def get_line_intersection(self,line1,line2):
        'Return (x,y) intersection of two lines given as (rho,theta)'
        rho1, theta1 = line1
        rho2, theta2 = line2
        a1 = np.cos(theta1)
        a2 = np.sin(theta1)
        b1 = np.cos(theta2)
        b2 = np.sin(theta2)
        d = a1*b2 - a2*b1
        if d==0:
            return (0,0)
        x = (rho1*b2-rho2*a2) / d
        y = (-rho1*b1+rho2*a1) / d
        return (x,y)
    
    def distance(self,pt1,pt2):
        'Returns euclidean distance between two points (x,y)'
        return ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5
    
    def is_not_convex(self,*pts):
        'Return true if points form convex shape'
        N = len(pts)
        prev, curr = 0, 0
        for i in range(N):
            temp = [pts[i],pts[(i+1)%N],pts[(i+2)%N]]
            curr = self.cross_product(temp)
            if curr != 0:
                if curr * prev < 0:
                    return True
                else:
                    prev = curr
        return False
    
    def cross_product(self,A):
        'Returns cross of 2 vectors in R2'
        X1 = (A[1][0] - A[0][0])
        Y1 = (A[1][1] - A[0][1])
        X2 = (A[2][0] - A[0][0])
        Y2 = (A[2][1] - A[0][1])
        return (X1 * Y2 - Y1 * X2)

    def regress_box_boundary(self,pts_src,delta=range(1,40)):
        '''Adjust box boundary to get better homography
        @Returns True is box boundary was adjusted and False otherwise'''
        prev_good = self.evaluate_homography(pts_src,self.BOX_BOUNDARY)
        box_bounds = []
        for i in [0]:
            for j in [-1,1]:
                for d in delta:
                    # copy = self.BOX_BOUNDARY.copy()
                    # copy[i:i+2,0] += delta*j
                    # box_bounds.append(copy)
                    copy = self.BOX_BOUNDARY.copy()
                    copy[[i-1,i],1] += d*j
                    box_bounds.append(copy)

        max_good = 0
        max_index = 0
        for i in range(len(box_bounds)):
            good = self.evaluate_homography(pts_src,box_bounds[i])
            if good > max_good:
                max_good = good
                max_index = i
        
        if max_good <= prev_good:
            return False
        else:
            self.BOX_BOUNDARY = box_bounds[max_index]
            return True
    
    def fine_regress_box_boundary(self,pts_src,delta=range(1,5)):
        '''Adjust box boundary finely to get better homography
        @Returns True is box boundary was adjusted and False otherwise'''
        prev_good = self.evaluate_homography(pts_src,self.BOX_BOUNDARY)
        box_bounds = []
        for i in [0,1,2,3]:
            for j in [0,1]:
                for k in [-1,1]:
                    for d in delta:
                        copy = self.BOX_BOUNDARY.copy()
                        copy[i,j] += d*k
                        box_bounds.append(copy)

        max_good = 0
        max_index = 0
        for i in range(len(box_bounds)):
            good = self.evaluate_homography(pts_src,box_bounds[i])
            if good > max_good:
                max_good = good
                max_index = i
        
        if max_good <= prev_good:
            return False
        else:
            self.BOX_BOUNDARY = box_bounds[max_index]
            return True
    
    def apply_hough(self,img:np.ndarray, lines:list):
        'Return image with hough transform lines drawn on top'
        out = img.copy()
        for line in lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a*rho, b*rho
            x1, y1 = int(x0 + 2000*(-b)), int(y0 + 2000*(a))
            x2, y2 = int(x0 - 2000*(-b)), int(y0 - 2000*(a)) 
            cv.line(out,(x1,y1),(x2,y2),[0,0,255])
        return out
    
    def transform_point(self,x:float,y:float):
        'Applies self.homography to point (x,y) and returns transformed point (x",y")'
        point = np.array([[x, y]], dtype=np.float32)

        # Reshape the point into a 1x1x2 array (required by cv2.perspectiveTransform())
        point = point.reshape((1, 1, 2))

        # Apply the homography to the point
        transformed_point = cv.perspectiveTransform(point, self.homography)

        # Extract the transformed coordinates as a tuple (tx, ty)
        tx, ty = transformed_point[0, 0]

        return tx, ty

    def apply_gray_homography(self,im_src:np.ndarray, pts_src:list, pts_dst=None, or_mask=False):
        '''
        Return warped image given list of four pts 
        @Preconditions: im_src is grayscale image of masked edges
        src_pts: list of fours (x,y)* starting at back right corner of box and looping around counterclockwise
        or_mask: lets us see all parts of both truth map and homographied image 
        '''
        im_dst = self.true_map.copy()
        if pts_dst is None:
            pts_dst = self.BOX_BOUNDARY
        pts_src = np.array(pts_src)
        h, _ = cv.findHomography(pts_src,pts_dst)
        im_out = cv.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
        if or_mask:
            return cv.bitwise_or(im_out,self.invert_grayscale(im_dst))
        else:
            return cv.bitwise_and(im_out,self.invert_grayscale(im_dst))

    def apply_bgr_homography(self,im_src:np.ndarray, pts_src:list):
        '''
        Return warped bgr image given list of four pts 
        @Preconditions: im_src is bgr image of court
        src_pts: list of fours (x,y)* starting at back right corner of box and looping around counterclockwise
        '''
        im_dst = self.true_map.copy()
        pts_dst = self.BOX_BOUNDARY
        pts_src = np.array(pts_src)
        h, _ = cv.findHomography(pts_src,pts_dst)
        im_out = cv.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
        return cv.bitwise_or(im_out,cv.cvtColor(self.invert_grayscale(im_dst),cv.COLOR_GRAY2BGR))

    def convert_to_hsv(self,source_image:str, alt:np.ndarray=None):
        'Helper function converts bgr image to hsv and separates into channels'
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

    def convert_to_ycrcb(self,source_image:str, alt:np.ndarray=None):
        'Helper function converts bgr image to ycrcb and separates into channels'
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

    def invert_grayscale(self,gray_img:np.ndarray):
        'Inverts grayscale images'
        ret = gray_img.copy()
        ret[ret[:,:] >= 128] = 128
        ret[ret[:,:] < 128] = 255
        ret[ret[:,:] == 128] = 0
        return ret

    def test_many_bins(self,img:np.ndarray, bgr:np.ndarray, bins:list, iterations:int=6):
        'Helper function to see what colors are in the bins'
        for i in range(max(iterations, len(bins))):
            mask = self.get_mask(img, bins[i])
            masked = self.apply_mask(bgr,mask)
            cv.imshow('Bin Level ' + str(i+1), masked)

    def test_many_canny(self,gray_img:np.ndarray, mask:np.ndarray, grid:list):
        'Helper function to see different threshold levels of canny edge detection'
        for one in grid:
            for two in grid:
                if one > two:
                    continue
                canny = self.get_canny(gray_img, threshold1=one, threshold2=two)
                masked_canny = self.apply_mask(canny, mask)
                cv.imshow(str(one)+' by '+str(two), masked_canny) 

    def test_many_hough(self,gray_img:np.ndarray, canny:np.ndarray, grid:list):
        'Helper function to test many hough lines of different thresholds'
        for rho in grid[0]:
            for theta in grid[1]:
                for threshold in grid[2]:
                    lines = self.get_hough(canny, rho, theta, threshold)
                    hough = self.apply_hough(gray_img, lines)
                    cv.imshow(str(rho)+' by '+str(round(theta,3))+' by '+str(threshold), hough)

if __name__ == '__main__':
    video_path = os.path.join('court-detection','temp','demo.mov')
    render = Render(video_path=video_path,display_images=True)

    # filename = os.path.join('court-detection','temp','point.mp4')
    # render.render_video(states,players,filename)