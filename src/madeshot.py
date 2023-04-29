#!/usr/bin/env python3
import pandas as pd


def new_rim(file_path):
    """
    Accepts a strongsort text file as an input, and outputs the coordinates of 
    the original rim and the new box above it. Assumption: the coordinates of 
    the rim do not move throughout the video. 
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            lst = [int(i) for i in line.split()]
            # rim = 2
            if lst[1] == 2:
                left = lst[3]
                top = lst[4]
                width = lst[5]
                height = lst[6]
    top_box = left, top + height, width, height
    rim_box = left, top, width, height
    return top_box, rim_box


def madeshot(file_path):
    top, rim = new_rim(file_path)
    shots_made = []
    passed_top = False
    passed_rim = False
    top_collision_start = 0
    top_collision_end = 0
    rim_collision_start = 0
    rim_collision_end = 0
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            lst = [int(i) for i in line.split()]
            frame = lst[0]
            # ball = [Some Number]
            if lst[1] == 3:
                ball_x = lst[3] + lst[5]/2
                ball_y = lst[4] - lst[6]/2

                in_top_x = (ball_x >= top[0] and ball_x <= top[0] + top[2])
                in_top_y = (ball_y <= top[1] and ball_y >= top[1] - top[3])
                if not passed_top:
                    if in_top_x and in_top_y and (frame not in
                                                  range(top_collision_start,
                                                        top_collision_end)):
                        top_collision_start = frame
                        passed_top = True
                else:
                    if not in_top_x or not in_top_y:
                        top_collision_end = frame

                in_rim_x = (ball_x >= rim[0] and ball_x <= rim[0] + rim[2])
                in_rim_y = (ball_y <= rim[1] and ball_y >= rim[1] - rim[3])
                if not passed_rim:
                    if in_rim_x and in_rim_y and (frame not in
                                                  range(rim_collision_start,
                                                        rim_collision_end)):
                        rim_collision_start = frame
                        passed_rim = True
                else:
                    if not in_rim_x or not in_rim_y:
                        rim_collision_end = frame

                if passed_top and passed_rim:
                    shots_made.append((top_collision_start, rim_collision_end))
                    passed_top = False
                    passed_rim = False
                    top_collision_start = 0
                    top_collision_end = 0
                    rim_collision_start = 0
                    rim_collision_end = 0

    return shots_made

print(new_rim('/data/test2.txt'))
print(madeshot('/data/test2.txt'))