from state import GameState, ShotAttempt, Box, Interval
from sklearn.linear_model import LinearRegression
import numpy as np


class ShotFrame:
    def __init__(self):
        self.top: bool = False
        self.rim: bool = False


def detect_shot(state: GameState, inte: Interval, window: int) -> ShotAttempt:
    """
    Returns a ShotAttempt analyzing frames along the interval
        window: margin of error in frames from top to rim box
    """
    sa = ShotAttempt(inte.playerid, inte.start, inte.end)
    sfs: list[ShotFrame] = []


    # ==========IMPLEMENTATION OF LINEAR REGRESSION METHOD (NEW)==========
    # get the ball center positions for each frame
    ball_positions = [(i, (state.frames[i].ball.box.xmin + state.frames[i].ball.box.xmax) / 2, (state.frames[i].ball.box.ymin + state.frames[i].ball.box.ymax) / 2) for i in range(sa.start, sa.end + 1)]

    if len(ball_positions) < 2:
        return sa
    

    # perform linear regression
    x = np.array([[pos[1] for pos in ball_positions]])
    y = np.array([[pos[2] for pos in ball_positions]])
    model = LinearRegression().fit(x, y)

    # use the model to predict the y-position of the ball at the rim
    rim_y = model.predict([[(state.frames[sa.end].rim.xmin + state.frames[sa.end].rim.xmax) / 2]])[0]
    true_y = (state.frames[sa.end].rim.ymin + state.frames[sa.end].rim.ymax) / 2

    # check if predicted y-coord is within hoop's y-range
    if rim_y >= true_y - window and rim_y <= true_y + window:
        sa.made = True

        # find the frame where the ball is at the rim
        for i in range(sa.start, sa.end + 1):
            ball = state.frames[i].ball
            if ball is not None:
                if ball.box.ymin <= rim_y and ball.box.ymax >= rim_y:
                    sa.frame = i
                    break
                
    else:
        sa.made = False
        sa.frame = sa.end

    return sa

    # ==========IMPLEMENTATION OF WINDOW METHOD (OLD)==========

    for i in range(sa.start, sa.end + 1):  # construct sfs of shot frames


        rim = state.frames[i].rim
        h = rim.ymax - rim.ymin
        top = Box(rim.xmin, rim.ymin - h, rim.xmax, rim.ymax - h)  # top rim box
        ball = state.frames[i].ball

        sf = ShotFrame()
        if ball is None or rim is None:
            pass
        else:
            if ball.box.intersects(top):
                sf.top = True
            if rim.contains(ball.box):
                sf.rim = True
        sfs.append(sf)

    r = 0  # freq rim intersects in window
    for i in range(len(sfs) - window):
        if sfs[i + window].rim:
            r += 1  # add to window
        if sfs[i].top and r > 0:
            sa.made = True  # made shot detected
            sa.frame = i + sa.start
        if sfs[i].rim:
            r -= 1  # remove from window

    return sa


def shots(state: GameState, window: int):
    """
    Calculate shots throughout game
        window: margin of error in frames from top to rim box
    Assumption:
        shots are counted as breaks between possesssions
    """
    # Create intervals of shots
    shots: list[Interval] = []
    poss = state.possessions
    for i in range(len(poss) - 1):
        p1 = poss[i]
        p2 = poss[i + 1]
        shots.append(Interval(p1.playerid, p1.end, p2.start))

    for inte in shots:
        sa: ShotAttempt = detect_shot(state, inte, window=window)
        state.shot_attempts.append(sa)

    state.populate_players_stats() # populate players stats
    state.populate_team_stats() # populate team stats
