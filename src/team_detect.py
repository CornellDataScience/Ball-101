def curr_possession(players, ball):
    '''
    Input:
        players: dictionary of players
        ball: dictionary containing coords of the ball
    Output:
        possession_list [list]: list of player ids in the order of ball possession throughout the video
    '''
    player_ids = players.keys()
    maxArea = 0
    maxPlayer = None
    bxmin = ball.get("xmin")
    bxmax = ball.get("xmax")
    bymin = ball.get("ymin")
    bymax = ball.get("ymax")
    for player in player_ids:
        pcoords = players.get(player)
        currxmin = max(pcoords.get("xmin"), bxmin)
        currxmax = min(pcoords.get("xmax"), bxmax)
        currymin = max(pcoords.get("ymin"), bymin)
        currymax = min(pcoords.get("ymax"), bymax)
        area = (currxmax - currxmin) * (currymax - currymin)
        if area > maxArea:
            maxArea = area
            maxPlayer = player
    return maxPlayer

def possession_list(state, thresh=20):
    '''
    Input:
        state: a StatState class that holds all sorts of information on the video
        thresh: number of frames for ball overlapping with player in order to count as possession
    Output:
        possession_list [list]: list of player ids in the order of ball possession throughout the video
    '''
    # list of frames where a frame is a dictionary with key classes and values xmin, ymin, xmax, ymax
    # for key = players value has the format {players: {playerid1: {xmin: val, ymin: val, xmax: val, ymax: val}}}
    states = state.states
    counter = 0
    current_player = None
    pos_lst = []
    for frame in states:
        if frame.get("ball") == None:
            continue
        poss = curr_possession(frame.get("players"), frame.get("ball"))

        if poss == None:
            continue
        elif poss == current_player:
            counter += 1
        else:
            current_player = poss
            counter = 0

        if counter >= thresh:
            pos_lst.append(poss)
    return pos_lst

def connections(pos_lst, players):
    connects = dict()
    for i in range(0, len(players)):
        for j in range(0, len(players)):
            if i == j:
                continue
            name = players[i] + players[j]
            connects.update({name: 0})
    curr = pos_lst[0]
    for i in range(1, len(pos_lst)):
        name = curr + pos_lst[i]
        connects[name] += 1
    return connects

def possible_teams(players):
    numpeople = len(players)
    pplperteam = numpeople/2
    acc = []
    def permutation(i, t):
        if i >= numpeople:
            return
        if len(t) == pplperteam:
            acc.append((t, (set(players) - set(t))))
        else:
            permutation(i+1, t.copy())
            t.add(players[i])
            permutation(i+1, t.copy())
    permutation(0, set())
    return acc

def team_split(state):
    '''
    Input:
        state: a StatState class that holds all sorts of information on the video
    Output:
        state: updated version of the input
    '''
    # list of player ids
    players = state.players
    # number of people total
    numpeople = len(players)
    pos_lst = possession_list(state, players)
    state.possession_list = pos_lst
    connects = connections(pos_lst, players)
    teams = possible_teams(players, numpeople)
    bestteam = None
    minCount = 0
    for team in teams:
        count = 0
        team1 = list(team[0])
        team2 = list(team[1])
        for player1 in team1:
            for player2 in team2:
                count += connects[player1 + player2]
                count += connects[player2 + player1]
        if count < minCount:
            minCount = count
            bestteam = team
    state.team1 = bestteam[0]
    state.team2 = bestteam[1]
    return state
