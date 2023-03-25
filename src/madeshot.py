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
            if lst[1] == 2:
                left = lst[3]
                top = lst[4]
                width = lst[5]
                height = lst[6]
    lower_box = left, top, width, height
    upper_box = left, top + height, width, height
    return lower_box, upper_box


def madeshot(file_path):
    lower, upper = new_rim(file_path)
    start_shot = 0
    end_shot = 0
    """
    To be implemented with singly linked list.
    """
