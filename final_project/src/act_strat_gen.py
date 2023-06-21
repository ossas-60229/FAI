import random
import json
from my_utils.card_funcs import gen_deck, get_all_combinations, get_all_possible_cards
from my_utils.card import Card
from my_utils.hand_evaluator import HandEvaluator
from my_utils.Mygame import Game
import numpy as np


fuck_d = dict()


# get a hole card grades
def hand_eval(hole_card, community_cards, all_hole):
    hole_hand = list(map(Card.from_str, hole_card))
    community_hand = list(map(Card.from_str, community_cards))
    return HandEvaluator().eval_hand(hole_hand, community_hand)


def win_rate(hole_card, community_cards, all_hole):
    win = lose = tie = 0
    my_val = hand_eval(hole_card, community_cards, all_hole)
    for hole in all_hole:
        if hole[0] in hole_card + community_cards:
            continue
        if hole[1] in hole_card + community_cards:
            continue
        opp_val = hand_eval(hole, community_cards, all_hole)
        if my_val > opp_val:
            win += 1
        elif my_val == opp_val:
            tie += 1
        else:
            lose += 1
    return (win + (tie) / 2) / (win + tie + lose)



def sort_card_strs(card_strs):
    SUIT_MAP = {2: 0, 4: 1, 8: 2, 16: 3}
    card_list = list(map(Card.from_str, card_strs))
    card_by_suit = [[] for _ in range(4)]
    for card in card_list:
        card_by_suit[SUIT_MAP[card.suit]].append(card)
    for lt in card_by_suit:
        lt.sort(key=lambda x: x.rank)
    ret_list = []
    for i in range(4):
        for card in card_by_suit[i]:
            ret_list.append(card.__str__())
    return tuple(ret_list)


def encode_hole_card(hole_cards):
    card_list = list(map(Card.from_str, hole_cards))
    suf = ""
    if card_list[0].suit == card_list[1].suit:
        suf = "s"
    elif card_list[0].rank == card_list[1].rank:
        suf = ""
    else:
        suf = "d"
    card_list.sort(key=lambda x: x.rank)
    ret_str = Card.RANK_MAP[card_list[0].rank] + Card.RANK_MAP[card_list[1].rank] + suf
    return ret_str

# generate strategy
strat_map = dict()
digit = 100
param = dict()
for i in range(4):
    for j in range(digit):
        for k in range(2):
            key = "%d_%d_%d" % (i, j, k)
            # (turn, rate, position)
            # i-th turn and win rate j*0.001
            if j > digit*0.9:
                if k == 1:
                    strat_map[key] = (0.85,0.1,0.05)
                else:
                    strat_map[key] = (0,1,0)
                # ratio of min raise
            elif j >= digit*0.4:
                strat_map[key] = (0.15,0.8,0.05)
            else:
                strat_map[key] = (0.05,0.05,0.9)
    #(action, default)
#(turn, win rate, position(fore or back), ratio of call bet and whole pot)
param["digit"] = digit
param["keys"] = ["turn", "win_rate", "position"]
root = "./strat/"
import os
if not os.path.exists(root):
    os.makedirs(root)
filname = "blank_strat.json"
param_name = "blank_param.json"
with open(root+filname, "w") as f:
    json.dump(strat_map, f, indent=2)
with open(root+param_name, "w") as f:
    json.dump(param, f, indent=2)

"""
cards = get_all_possible_cards()
hole_cards = get_all_combinations(cards, 2)
community_cards = get_all_combinations(cards, 3)
hand_winrate_dict = dict()
flop_winrate_dict = dict()
print("fucking start\n\n\n\n")
comb = 0
for hand in hole_cards:
    key_1 = encode_hole_card(hand)
    if key_1 in hand_winrate_dict:
        print("fuck continu")
        continue
    total = step = 0
    for flop in community_cards:
        if hand[0] in flop or hand[1] in flop:
            continue
        comb += 1
        print(comb)
        hand2 = list(map(Card.from_str, hand))
        flop2 = list(map(Card.from_str, flop))
        key_2 = str(HandEvaluator().eval_hand(hand2, flop2))
        tmp = 0
        if key_1+key_2 in flop_winrate_dict:
            tmp = flop_winrate_dict[key_1+key_2]
        else:
            tmp = win_rate(hand, flop, hole_cards)
            flop_winrate_dict[key_1+key_2] = tmp
        step += 1
        total += tmp
    hand_winrate_dict[key_1] = total / step
        
        
file_name1 = "hand_winrate_flop.json"
file_name2 = "flop_winrate.json"
with open(file_name1, "w") as f:
    json.dump(hand_winrate_dict, f)
with open(file_name2, "w") as f:
    json.dump(flop_winrate_dict, f)
for i in range(4):
    for j in range(digit):
        for k in range(2):
            for q in range(digit):
                key = "%d_%d_%d_%d" % (i, j, k, q)
                # (turn, rate, position)
                # i-th turn and win rate j*0.001
                if j > digit*0.7:
                    if k == 1:
                        strat_map[key] = (1,0)
                    else:
                        strat_map[key] = (2,2)
                    # ratio of min raise
                elif j >= digit*0.4:
                    strat_map[key] = (1,0)
                else:
                    strat_map[key] = (0,0)
        #(action, default)
#(turn, win rate, position(fore or back), ratio of call bet and whole pot)
    """