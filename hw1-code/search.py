
# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
##
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

# """
# This is the main entry point for HW1. You should only modify code
# within this file -- the unrevised staff files will be used for all other
# files and classes when code is run, so be careful to not modify anything else.
# """
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)
# """"""


#import copy
#import sys
#import heapq
#import math
# def search(maze, searchMethod):
# return {
# "bfs": bfs,
# "astar": astar,
# "astar_corner": astar_corner,
# "astar_multi": astar_multi,
# "fast": fast,
# }.get(searchMethod)(maze)


# def bfs(maze):
# """
# Runs BFS for part 1 of the assignment.

# @param maze: The maze to execute the search on.

# @return path: a list of tuples containing the coordinates of each state in the computed path
# """
# TODO: Write your code here
#que = []
#start = maze.getStart()
# que.append(start)
#done = set()
#obj = maze.getObjectives()
#table = dict()
# while len(que) != 0:
#tmp = que.pop(0)
# if tmp in obj:
# obj.remove(tmp)
# if len(obj) == 0:
# break
#n_list = maze.getNeighbors(tmp[0], tmp[1])
# for nb in n_list:
# if (not nb in done):
# que.append(nb)
#table[nb] = tmp
# done.add(nb)
# n_list.clear()
#ret = [tmp]
# if (len(obj) == 0):
# while tmp != start:
# ret.append(table[tmp])
#tmp = ret[-1]
# ret.append(start)
# ret.reverse()
# return ret


# def astar(maze):
# """
# Runs A star for part 1 of the assignment.

# @param maze: The maze to execute the search on.

# @return path: a list of tuples containing the coordinates of each state in the computed path
# """
# TODO: Write your code here
# def heuristic(dot1, dot2):
# return abs(dot1[0] - dot2[0]) + abs(dot1[1] - dot2[1])
#p_que = []
#start = maze.getStart()
#done = set()
#obj = maze.getObjectives()
#heapq.heappush(p_que, (heuristic(start, obj[0]), 0, start[0], start[1]))
#table = dict()
# while len(p_que) != 0:
#temp = heapq.heappop(p_que)
#tmp = (temp[2], temp[3])
# if tmp in obj:
# obj.remove(tmp)
# if len(obj) == 0:
# break
#n_list = maze.getNeighbors(tmp[0], tmp[1])
# for nb in n_list:
# if (not nb in done):
# heapq.heappush(p_que, (heuristic(
# nb, obj[0]) + temp[1] + 1, temp[1] + 1, nb[0], nb[1]))
#table[nb] = tmp
# done.add(nb)
# n_list.clear()
#ret = [tmp]
# if (len(obj) == 0):
# while tmp != start:
# ret.append(table[tmp])
#tmp = ret[-1]
# ret.append(start)
# ret.reverse()
# return ret


# def astar_corner(maze):
# """
# Runs A star for part 2 of the assignment in the case where there are four corner objectives.

# @param maze: The maze to execute the search on.

# @return path: a list of tuples containing the coordinates of each state in the computed path
# """
# TODO: Write your code here
# def manhatton(dot1, dot2):
# return abs(dot1[0] - dot2[0]) + abs(dot1[1] - dot2[1])

# def heuristic(dot1, goal_list):
# if len(goal_list) == 0:
# return 0
#minh = manhatton(dot1, goal_list[0])
# for goal in goal_list:
#minh = min(minh, manhatton(dot1, goal))
# return minh

# def shortest_path(start, goal):
#p_que = []
#done = set()
#obj = [goal]
#heapq.heappush(p_que, (heuristic(start, obj), 0, start[0], start[1]))
#table = dict()
# while len(p_que) != 0:
#temp = heapq.heappop(p_que)
#tmp = (temp[2], temp[3])
# if tmp in obj:
# obj.remove(tmp)
# break
#n_list = maze.getNeighbors(tmp[0], tmp[1])
# for nb in n_list:
# if (not nb in done):
# heapq.heappush(p_que, (heuristic(
# nb, obj) + temp[1] + 1, temp[1] + 1, nb[0], nb[1]))
#table[nb] = tmp
# done.add(nb)
# n_list.clear()
#ret = [tmp]
# if (len(obj) == 0):
# while tmp != start:
# ret.append(table[tmp])
#tmp = ret[-1]
# ret.append(start)
# ret.reverse()
# return len(ret)
# apply with DP

# def mst_heur(pos, goals, edges, heur_table):
# def mst_caculator(vertices, edges):
# if (len(vertices) == 0):
# return 0
#vs = list(vertices)
#cur = [vertices[0]]
# vs.remove(vertices[0])
#ret = 0
# while (len(cur) < len(vertices)):
#mst_paths = []
# for now_v in cur:
#min_val = sys.maxsize
#min_nod = None
# for unv_v in vs:
# if min_val > edges[(now_v, unv_v)]:
#min_val = edges[(now_v, unv_v)]
#min_nod = unv_v
#mst_paths.append((min_val, min_nod))
#min_path = min(mst_paths)
# vs.remove(min_path[1])
#ret += min_path[0]
# cur.append(min_path[1])
# return ret
#heur = 0
#mah_dist = heuristic(pos, goals)
# if (goals in heur_table):
#heur = heur_table[goals]
# else:
#heur = mst_caculator(goals, edges)
#heur_table[goals] = heur
# return mah_dist + heur
#start = maze.getStart()
#objs = maze.getObjectives()
#len_bt_obj = dict()
# for a_obj in objs:
# for b_obj in objs:
# if (b_obj, a_obj) in len_bt_obj:
#len_bt_obj[(a_obj, b_obj)] = len_bt_obj[(b_obj, a_obj)]
# elif a_obj != b_obj:
#len_bt_obj[(a_obj, b_obj)] = shortest_path(a_obj, b_obj)
# heap node = (heur, pos, tuple(goals))
#que = []
#heur_table = dict()
#dist_table = dict()
#path_table = dict()
#dist_table[(start, tuple(objs))] = 0
# tnode = (mst_heur(start, tuple(objs),
# len_bt_obj, heur_table), start, tuple(objs))
#heapq.heappush(que, tnode)
# while len(que) != 0:
#now_node = heapq.heappop(que)
# if (len(now_node[2]) == 0):
# break
#now_cor = now_node[1]
#nbs = maze.getNeighbors(now_cor[0], now_cor[1])
# for nb in nbs:
#tmp_goals = list(tuple(now_node[2]))
# if nb in tmp_goals:
# tmp_goals.remove(nb)
#the_goals = tuple(tmp_goals)
#nb_node = (nb, tuple(the_goals))
#tmp_node = (now_cor, tuple(now_node[2]))
# if nb_node in dist_table and dist_table[nb_node] <= dist_table[tmp_node] + 1:
# continue
# else:
#dist_table[nb_node] = dist_table[tmp_node] + 1
#path_table[nb_node] = tmp_node
# cmp = dist_table[nb_node] + \
#mst_heur(nb, tuple(the_goals), len_bt_obj, heur_table)
#heapq.heappush(que, (cmp, nb, tuple(the_goals)))
#tempnode = (now_node[1], tuple(now_node[2]))
#ret = [tempnode]
# while ret[-1][0] != start or len(objs) != 0:
# if (ret[-1][0] in objs):
# objs.remove(ret[-1][0])
# ret.append(path_table[ret[-1]])
# ret.reverse()
#fret = []
# for fuck in ret:
# fret.append(fuck[0])
# return fret


# def astar_multi(maze):
# """
# Runs A star for part 3 of the assignment in the case where there are
# multiple objectives.

# @param maze: The maze to execute the search on.

# @return path: a list of tuples containing the coordinates of each state in the computed path
# """
# TODO: Write your code here
# def manhatton(dot1, dot2):
# return abs(dot1[0] - dot2[0]) + abs(dot1[1] - dot2[1])

# def heuristic(dot1, goal_list):
# if len(goal_list) == 0:
# return 0
#minh = manhatton(dot1, goal_list[0])
# for goal in goal_list:
#minh = min(minh, manhatton(dot1, goal))
# return minh

# def shortest_path(start, goal):
#p_que = []
#done = set()
#obj = [goal]
#heapq.heappush(p_que, (heuristic(start, obj), 0, start[0], start[1]))
#table = dict()
# while len(p_que) != 0:
#temp = heapq.heappop(p_que)
#tmp = (temp[2], temp[3])
# if tmp in obj:
# obj.remove(tmp)
# break
#n_list = maze.getNeighbors(tmp[0], tmp[1])
# for nb in n_list:
# if (not nb in done):
# heapq.heappush(p_que, (heuristic(
# nb, obj) + temp[1] + 1, temp[1] + 1, nb[0], nb[1]))
#table[nb] = tmp
# done.add(nb)
# n_list.clear()
#ret = [tmp]
# if (len(obj) == 0):
# while tmp != start:
# ret.append(table[tmp])
#tmp = ret[-1]
# ret.append(start)
# ret.reverse()
# return len(ret)
# apply with DP

# def mst_heur(pos, goals, edges, heur_table):
# def mst_caculator(vertices, edges):
# if (len(vertices) == 0):
# return 0
#vs = list(vertices)
#cur = [vertices[0]]
# vs.remove(vertices[0])
#ret = 0
# while (len(cur) < len(vertices)):
#mst_paths = []
# for now_v in cur:
#min_val = sys.maxsize
#min_nod = None
# for unv_v in vs:
# if min_val > edges[(now_v, unv_v)]:
#min_val = edges[(now_v, unv_v)]
#min_nod = unv_v
#mst_paths.append((min_val, min_nod))
#min_path = min(mst_paths)
# vs.remove(min_path[1])
#ret += min_path[0]
# cur.append(min_path[1])
# return ret
#heur = 0
#mah_dist = heuristic(pos, goals)
# if (goals in heur_table):
#heur = heur_table[goals]
# else:
#heur = mst_caculator(goals, edges)
#heur_table[goals] = heur
# return mah_dist + heur

#start = maze.getStart()
#objs = maze.getObjectives()
#len_bt_obj = dict()
# for a_obj in objs:
# for b_obj in objs:
# if (b_obj, a_obj) in len_bt_obj:
#len_bt_obj[(a_obj, b_obj)] = len_bt_obj[(b_obj, a_obj)]
# elif a_obj != b_obj:
#len_bt_obj[(a_obj, b_obj)] = shortest_path(a_obj, b_obj)
# heap node = (heur, pos, tuple(goals))
#que = []
#heur_table = dict()
#dist_table = dict()
#path_table = dict()
#dist_table[(start, tuple(objs))] = 0
# tnode = (mst_heur(start, tuple(objs),
# len_bt_obj, heur_table), start, tuple(objs))
#heapq.heappush(que, tnode)
# while len(que) != 0:
#now_node = heapq.heappop(que)
# if (len(now_node[2]) == 0):
# break
#now_cor = now_node[1]
#nbs = maze.getNeighbors(now_cor[0], now_cor[1])
# for nb in nbs:
#tmp_goals = copy.deepcopy(list(tuple(now_node[2])))
# if nb in tmp_goals:
# tmp_goals.remove(nb)
#the_goals = tuple(tmp_goals)
#nb_node = (nb, tuple(the_goals))
#tmp_node = (now_cor, tuple(now_node[2]))
# if nb_node in dist_table and dist_table[nb_node] <= dist_table[tmp_node] + 1:
# continue
# else:
#dist_table[nb_node] = dist_table[tmp_node] + 1
#path_table[nb_node] = tmp_node
# cmp = dist_table[nb_node] + \
#mst_heur(nb, tuple(the_goals), len_bt_obj, heur_table)
#heapq.heappush(que, (cmp, nb, tuple(the_goals)))
#tempnode = (now_node[1], tuple(now_node[2]))
#ret = [tempnode]
# while ret[-1][0] != start or len(objs) != 0:
# if (ret[-1][0] in objs):
# objs.remove(ret[-1][0])
# ret.append(path_table[ret[-1]])
# ret.reverse()
#fret = []
# for fuck in ret:
# fret.append(fuck[0])
# return fret


# def fast(maze):
# """
# Runs suboptimal search algorithm for part 4.

# @param maze: The maze to execute the search on.

# @return path: a list of tuples containing the coordinates of each state in the computed path
# """
# TODO: Write your code here
# def manhatton(dot1, dot2):
# return abs(dot1[0] - dot2[0]) + abs(dot1[1] - dot2[1])

# def heuristic(dot1, goal_list):
# if len(goal_list) == 0:
# return 0
#minh = manhatton(dot1, goal_list[0])
# for goal in goal_list:
#minh = min(minh, manhatton(dot1, goal))
# return minh

# def close_one(pos, goals):
# if (len(goals) == 0):
# return pos
#ret = goals[0]
#min_dist = manhatton(pos, ret)
# for goal in goals:
#tmp = manhatton(pos, goal)
# if (tmp < min_dist):
#min_dist = tmp
#ret = goal
# return ret

# def shortest_path(start, goal):
#p_que = []
#done = set()
#obj = [goal]
#heapq.heappush(p_que, (heuristic(start, obj), 0, start[0], start[1]))
#table = dict()
# while len(p_que) != 0:
#temp = heapq.heappop(p_que)
#tmp = (temp[2], temp[3])
# if tmp in obj:
# obj.remove(tmp)
# break
#n_list = maze.getNeighbors(tmp[0], tmp[1])
# for nb in n_list:
# if (not nb in done):
# heapq.heappush(p_que, (heuristic(
# nb, obj) + temp[1] + 1, temp[1] + 1, nb[0], nb[1]))
#table[nb] = tmp
# done.add(nb)
# n_list.clear()
#ret = [tmp]
# if (len(obj) == 0):
# while tmp != start:
# ret.append(table[tmp])
#tmp = ret[-1]
# ret.append(start)
# ret.reverse()
# return ret
# apply with DP
#start = maze.getStart()
#objs = maze.getObjectives()
#que = []
#now_pos = start
#ret = []
#goal = objs[0]
# while (len(objs) > 0):
#goal = close_one(now_pos, objs)
# objs.remove(goal)
#tmp_path = shortest_path(now_pos, goal)
# tmp_path.remove(goal)
#ret += tmp_path
#now_pos = goal
# ret.append(goal)
# return ret
# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

import copy
import sys
import heapq
import math


def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    que = []
    start = maze.getStart()
    que.append(start)
    done = set()
    obj = maze.getObjectives()
    table = dict()
    while len(que) != 0:
        tmp = que.pop(0)
        if tmp in obj:
            obj.remove(tmp)
        if len(obj) == 0:
            break
        n_list = maze.getNeighbors(tmp[0], tmp[1])
        for nb in n_list:
            if (not nb in done):
                que.append(nb)
                table[nb] = tmp
                done.add(nb)
        n_list.clear()
    ret = [tmp]
    if (len(obj) == 0):
        while tmp != start:
            ret.append(table[tmp])
            tmp = ret[-1]
        ret.append(start)
    ret.reverse()
    return ret


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    def heuristic(dot1, dot2):
        return abs(dot1[0] - dot2[0]) + abs(dot1[1] - dot2[1])
    p_que = []
    start = maze.getStart()
    done = set()
    obj = maze.getObjectives()
    heapq.heappush(p_que, (heuristic(start, obj[0]), 0, start[0], start[1]))
    table = dict()
    while len(p_que) != 0:
        temp = heapq.heappop(p_que)
        tmp = (temp[2], temp[3])
        if tmp in obj:
            obj.remove(tmp)
        if len(obj) == 0:
            break
        n_list = maze.getNeighbors(tmp[0], tmp[1])
        for nb in n_list:
            if (not nb in done):
                heapq.heappush(p_que, (heuristic(
                    nb, obj[0]) + temp[1] + 1, temp[1] + 1, nb[0], nb[1]))
                table[nb] = tmp
                done.add(nb)
        n_list.clear()
    ret = [tmp]
    if (len(obj) == 0):
        while tmp != start:
            ret.append(table[tmp])
            tmp = ret[-1]
        ret.append(start)
    ret.reverse()
    return ret


def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    def manhatton(dot1, dot2):
        return abs(dot1[0] - dot2[0]) + abs(dot1[1] - dot2[1])

    def heuristic(dot1, goal_list):
        if len(goal_list) == 0:
            return 0
        minh = manhatton(dot1, goal_list[0])
        for goal in goal_list:
            minh = min(minh, manhatton(dot1, goal))
        return minh

    def shortest_path(start, goal):
        p_que = []
        done = set()
        obj = [goal]
        heapq.heappush(p_que, (heuristic(start, obj), 0, start[0], start[1]))
        table = dict()
        while len(p_que) != 0:
            temp = heapq.heappop(p_que)
            tmp = (temp[2], temp[3])
            if tmp in obj:
                obj.remove(tmp)
                break
            n_list = maze.getNeighbors(tmp[0], tmp[1])
            for nb in n_list:
                if (not nb in done):
                    heapq.heappush(p_que, (heuristic(
                        nb, obj) + temp[1] + 1, temp[1] + 1, nb[0], nb[1]))
                    table[nb] = tmp
                    done.add(nb)
            n_list.clear()
        ret = [tmp]
        if (len(obj) == 0):
            while tmp != start:
                ret.append(table[tmp])
                tmp = ret[-1]
            ret.append(start)
        ret.reverse()
        return len(ret)
    # apply with DP

    def mst_heur(pos, goals, edges, heur_table):
        def mst_caculator(vertices, edges):
            if (len(vertices) == 0):
                return 0
            vs = list(vertices)
            cur = [vertices[0]]
            vs.remove(vertices[0])
            ret = 0
            while (len(cur) < len(vertices)):
                mst_paths = []
                for now_v in cur:
                    min_val = sys.maxsize
                    min_nod = None
                    for unv_v in vs:
                        if min_val > edges[(now_v, unv_v)]:
                            min_val = edges[(now_v, unv_v)]
                            min_nod = unv_v
                    mst_paths.append((min_val, min_nod))
                min_path = min(mst_paths)
                vs.remove(min_path[1])
                ret += min_path[0]
                cur.append(min_path[1])
            return ret
        heur = 0
        mah_dist = heuristic(pos, goals)
        if (goals in heur_table):
            heur = heur_table[goals]
        else:
            heur = mst_caculator(goals, edges)
            heur_table[goals] = heur
        return mah_dist + heur
    start = maze.getStart()
    objs = maze.getObjectives()
    len_bt_obj = dict()
    for a_obj in objs:
        for b_obj in objs:
            if (b_obj, a_obj) in len_bt_obj:
                len_bt_obj[(a_obj, b_obj)] = len_bt_obj[(b_obj, a_obj)]
            elif a_obj != b_obj:
                len_bt_obj[(a_obj, b_obj)] = shortest_path(a_obj, b_obj)
    # heap node = (heur, pos, tuple(goals))
    que = []
    heur_table = dict()
    dist_table = dict()
    path_table = dict()
    dist_table[(start, tuple(objs))] = 0
    tnode = (mst_heur(start, tuple(objs),
                      len_bt_obj, heur_table), start, tuple(objs))
    heapq.heappush(que, tnode)
    while len(que) != 0:
        now_node = heapq.heappop(que)
        if (len(now_node[2]) == 0):
            break
        now_cor = now_node[1]
        nbs = maze.getNeighbors(now_cor[0], now_cor[1])
        for nb in nbs:
            tmp_goals = list(tuple(now_node[2]))
            if nb in tmp_goals:
                tmp_goals.remove(nb)
            the_goals = tuple(tmp_goals)
            nb_node = (nb, tuple(the_goals))
            tmp_node = (now_cor, tuple(now_node[2]))
            if nb_node in dist_table and dist_table[nb_node] <= dist_table[tmp_node] + 1:
                continue
            else:
                dist_table[nb_node] = dist_table[tmp_node] + 1
                path_table[nb_node] = tmp_node
                cmp = dist_table[nb_node] + \
                    mst_heur(nb, tuple(the_goals), len_bt_obj, heur_table)
                heapq.heappush(que, (cmp, nb, tuple(the_goals)))
    tempnode = (now_node[1], tuple(now_node[2]))
    ret = [tempnode]
    while ret[-1][0] != start or len(objs) != 0:
        if (ret[-1][0] in objs):
            objs.remove(ret[-1][0])
        ret.append(path_table[ret[-1]])
    ret.reverse()
    fret = []
    for fuck in ret:
        fret.append(fuck[0])
    return fret


def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    def manhatton(dot1, dot2):
        return abs(dot1[0] - dot2[0]) + abs(dot1[1] - dot2[1])

    def heuristic(dot1, goal_list):
        if len(goal_list) == 0:
            return 0
        minh = manhatton(dot1, goal_list[0])
        for goal in goal_list:
            minh = min(minh, manhatton(dot1, goal))
        return minh

    def shortest_path(start, goal):
        p_que = []
        done = set()
        obj = [goal]
        heapq.heappush(p_que, (heuristic(start, obj), 0, start[0], start[1]))
        table = dict()
        while len(p_que) != 0:
            temp = heapq.heappop(p_que)
            tmp = (temp[2], temp[3])
            if tmp in obj:
                obj.remove(tmp)
                break
            n_list = maze.getNeighbors(tmp[0], tmp[1])
            for nb in n_list:
                if (not nb in done):
                    heapq.heappush(p_que, (heuristic(
                        nb, obj) + temp[1] + 1, temp[1] + 1, nb[0], nb[1]))
                    table[nb] = tmp
                    done.add(nb)
            n_list.clear()
        ret = [tmp]
        if (len(obj) == 0):
            while tmp != start:
                ret.append(table[tmp])
                tmp = ret[-1]
            ret.append(start)
        ret.reverse()
        return len(ret)
    # apply with DP

    def mst_heur(pos, goals, edges, heur_table):
        def mst_caculator(vertices, edges):
            if (len(vertices) == 0):
                return 0
            vs = list(vertices)
            cur = [vertices[0]]
            vs.remove(vertices[0])
            ret = 0
            while (len(cur) < len(vertices)):
                mst_paths = []
                for now_v in cur:
                    min_val = sys.maxsize
                    min_nod = None
                    for unv_v in vs:
                        if min_val > edges[(now_v, unv_v)]:
                            min_val = edges[(now_v, unv_v)]
                            min_nod = unv_v
                    mst_paths.append((min_val, min_nod))
                min_path = min(mst_paths)
                vs.remove(min_path[1])
                ret += min_path[0]
                cur.append(min_path[1])
            return ret
        heur = 0
        mah_dist = heuristic(pos, goals)
        if (goals in heur_table):
            heur = heur_table[goals]
        else:
            heur = mst_caculator(goals, edges)
            heur_table[goals] = heur
        return mah_dist + heur

    start = maze.getStart()
    objs = maze.getObjectives()
    len_bt_obj = dict()
    for a_obj in objs:
        for b_obj in objs:
            if (b_obj, a_obj) in len_bt_obj:
                len_bt_obj[(a_obj, b_obj)] = len_bt_obj[(b_obj, a_obj)]
            elif a_obj != b_obj:
                len_bt_obj[(a_obj, b_obj)] = shortest_path(a_obj, b_obj)
    # heap node = (heur, pos, tuple(goals))
    que = []
    heur_table = dict()
    dist_table = dict()
    path_table = dict()
    dist_table[(start, tuple(objs))] = 0
    tnode = (mst_heur(start, tuple(objs),
                      len_bt_obj, heur_table), start, tuple(objs))
    heapq.heappush(que, tnode)
    while len(que) != 0:
        now_node = heapq.heappop(que)
        if (len(now_node[2]) == 0):
            break
        now_cor = now_node[1]
        nbs = maze.getNeighbors(now_cor[0], now_cor[1])
        for nb in nbs:
            tmp_goals = copy.deepcopy(list(tuple(now_node[2])))
            if nb in tmp_goals:
                tmp_goals.remove(nb)
            the_goals = tuple(tmp_goals)
            nb_node = (nb, tuple(the_goals))
            tmp_node = (now_cor, tuple(now_node[2]))
            if nb_node in dist_table and dist_table[nb_node] <= dist_table[tmp_node] + 1:
                continue
            else:
                dist_table[nb_node] = dist_table[tmp_node] + 1
                path_table[nb_node] = tmp_node
                cmp = dist_table[nb_node] + \
                    mst_heur(nb, tuple(the_goals), len_bt_obj, heur_table)
                heapq.heappush(que, (cmp, nb, tuple(the_goals)))
    tempnode = (now_node[1], tuple(now_node[2]))
    ret = [tempnode]
    while ret[-1][0] != start or len(objs) != 0:
        if (ret[-1][0] in objs):
            objs.remove(ret[-1][0])
        ret.append(path_table[ret[-1]])
    ret.reverse()
    fret = []
    for fuck in ret:
        fret.append(fuck[0])
    return fret


def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    def manhatton(dot1, dot2):
        return abs(dot1[0] - dot2[0]) + abs(dot1[1] - dot2[1])

    def heuristic(dot1, goal_list):
        if len(goal_list) == 0:
            return 0
        minh = manhatton(dot1, goal_list[0])
        for goal in goal_list:
            minh = min(minh, manhatton(dot1, goal))
        return minh

    def close_one(pos, goals):
        if (len(goals) == 0):
            return pos
        ret = goals[0]
        min_dist = manhatton(pos, ret)
        for goal in goals:
            tmp = manhatton(pos, goal)
            if (tmp < min_dist):
                min_dist = tmp
                ret = goal
        return ret

    def shortest_path(start, goal):
        p_que = []
        done = set()
        obj = [goal]
        heapq.heappush(p_que, (heuristic(start, obj), 0, start[0], start[1]))
        table = dict()
        while len(p_que) != 0:
            temp = heapq.heappop(p_que)
            tmp = (temp[2], temp[3])
            if tmp in obj:
                obj.remove(tmp)
                break
            n_list = maze.getNeighbors(tmp[0], tmp[1])
            for nb in n_list:
                if (not nb in done):
                    heapq.heappush(p_que, (heuristic(
                        nb, obj) + temp[1] + 1, temp[1] + 1, nb[0], nb[1]))
                    table[nb] = tmp
                    done.add(nb)
            n_list.clear()
        ret = [tmp]
        if (len(obj) == 0):
            while tmp != start:
                ret.append(table[tmp])
                tmp = ret[-1]
            ret.append(start)
        ret.reverse()
        return ret
    # apply with DP
    start = maze.getStart()
    objs = maze.getObjectives()
    que = []
    now_pos = start
    ret = []
    goal = objs[0]
    while (len(objs) > 0):
        goal = close_one(now_pos, objs)
        objs.remove(goal)
        tmp_path = shortest_path(now_pos, goal)
        tmp_path.remove(goal)
        ret += tmp_path
        now_pos = goal
    ret.append(goal)
    return ret
