import random as rand
import json
from my_utils.card_funcs import gen_deck, get_all_combinations, get_all_possible_cards
from my_utils.card import Card
from my_utils.hand_evaluator import HandEvaluator
from my_utils.Mygame import Game
import numpy as np
import os
from agents import my_player


class CFR:
    all_cards = get_all_possible_cards()

    def __init__(self) -> None:
        self.game_state_rd = dict()
        self.path = "./strat/train0_strat.json"
        self.sig_digits = 100
        self.ante = 1
        self.strat = dict()
        self.all_holes = get_all_combinations(self.all_cards,2)

    def load_strat(self, path):
        self.path = path
        self.strat = json.load(open(path, "r"))
        for key, val in self.strat.items():
            self.game_state_rd[key] = Node(key=key, strat=val)

    @staticmethod
    def hand_eval(hole_card, community_cards, all_hole):
        #print("fuck hole_card\n\n\n",hole_card, " \n\n")
        hole_hand = list(map(Card.from_str, hole_card))
        community_hand = list(map(Card.from_str, community_cards))
        return HandEvaluator().eval_hand(hole_hand, community_hand)

    @staticmethod
    def win_rate(hole_card, community_cards, all_hole):
        win = lose = tie = 0
        rest = 5 - len(community_cards)
        dup = CFR.all_cards.copy()
        for card in hole_card:
            dup.remove(card)
        for card in community_cards:
            dup.remove(card)
        rd_com = rand.choices(dup, k=rest)
        my_val = CFR.hand_eval(hole_card, community_cards + rd_com, all_hole)
        # print("fuck rd_com\n\n", rd_com," \n\n")
        for hole in all_hole:
            if hole[0] in hole_card + community_cards + rd_com:
                continue
            if hole[1] in hole_card + community_cards + rd_com:
                continue
            opp_val = CFR.hand_eval(hole, community_cards + rd_com, all_hole)
            if my_val > opp_val:
                win += 1
            elif my_val == opp_val:
                tie += 1
            else:
                lose += 1
        return (win + (tie) / 2) / (win + tie + lose)

    def train(self, iterations, n_rounds, sb):
        self.small_blind = sb
        self.big_blind = sb * 2
        util = 0
        self.street = 2
        interval = 100
        step = 0
        for i in range(iterations):
            if i % interval == 0:
                print("iteration", i)
            p1_hole, p2_hole, community_cards = Game.deal_cards()
            p1_hole = [card.__str__() for card in p1_hole]
            p2_hole = [card.__str__() for card in p2_hole]
            community_cards = [card.__str__() for card in community_cards]
            history = list()
            step += 1
            pram_list = [
                history,
                p1_hole,
                p2_hole,
                community_cards,
                0,# pot,
                1,# prob1,
                1,# prob2,
                0,# bet1,
                0,# bet2,
                0,# call_amount
            ]
            util += self.cfr(params_list=pram_list.copy())

        return util / step

    def encode_condition(self, win_rate, street, position, call_amount, pot, num, cost):
        stt = 0
        if street == "preflop":
            stt = 0
        elif street == "flop":
            stt = 1
        elif street == "turn":
            stt = 2
        elif street == "river":
            stt = 3
        pos = position + 1 / num
        if pos > 0.5:
            pos = 1
        else:
            pos = 0
        rt = int(win_rate * self.sig_digits)
        if rt >= self.sig_digits:
            rt = self.sig_digits - 1
        rt %= self.sig_digits

        ret = "%d_%d_%d" % (stt, rt, pos)
        return ret

    TURN_MAP = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}

    def cfr(self, params_list):
        (
            history,
            p1_hole,
            p2_hole,
            community_cards,
            pot,
            prob1,
            prob2,
            bet1,
            bet2,
            call_amount,
        ) = params_list
        # start from preflop
        # 1. preflop
        num = 2
        cmt = []
        if len(history) % 2 == 1:
            # p2 do
            action = 0
            amt = 0
            if history[-1][0] == Game.FOLD:
                fuck = 0
                for act in history:
                    if act[0] == Game.RAISE:
                        fuck += act[1]
                return -fuck
            elif history[-1][0] == Game.CALL or history[-1][0] == Game.RAISE:
                if len(history) == 1:
                    # determine the winner
                    #lend = 3
                    lend =  2 + self.street if self.street >= 1 else 0
                    # win_r = self.win_rate(
                    # p1_hole, community_cards[0:lend], self.all_cards.copy()
                    # )
                    # key = self.encode_condition(
                    # win_r,
                    # self.TURN_MAP[self.street],
                    # 0,
                    # call_amount,
                    # pot,
                    # num,
                    # 0,
                    # )
                    p1_val = CFR.hand_eval(p1_hole, community_cards, None)
                    p2_val = CFR.hand_eval(p2_hole, community_cards, None)
                    bet2 = bet1
                    pot = bet1 + bet2
                    if p1_val == p2_val:
                        return (pot / 2) - bet1
                    elif p1_val > p2_val:
                        return bet2
                    else:
                        return -bet1
                else:
                    lend =  2 + self.street if self.street >= 1 else 0
                    win_r = self.win_rate(
                        p1_hole, community_cards[0:lend], self.all_holes.copy()
                    )
                    key = self.encode_condition(
                        win_r,
                        self.TURN_MAP[self.street],
                        0,
                        call_amount,
                        pot,
                        num,
                        0,
                    )
                    key_val = self.strat[key]
                    if key in self.game_state_rd:
                        node_t = self.game_state_rd[key]
                    else:
                        node_t = Node(key=key, strat=list(self.strat[key]).copy())
                        self.game_state_rd[key] = node_t
                    strategy = node_t.getStrategy(prob2)
                    actions = [Game.FOLD, Game.CALL, Game.RAISE]
                    amts = [0, call_amount, self.ante]
                    for i in range(3):
                        dup_hist = history.copy()
                        dup_hist.append((actions[i], amts[i]))
                        params_list = [
                            dup_hist,
                            p1_hole,
                            p2_hole,
                            community_cards,
                            pot,
                            prob1,
                            prob2 * key_val[i],
                            bet1,
                            bet2 + amts[i],
                            0,
                        ]

        if len(history) % 2 == 0:
            # p1 do
            if len(history) == 0:
                lend =  2 + self.street if self.street >= 1 else 0
                win_r = self.win_rate(
                    p1_hole, community_cards[0:lend], self.all_holes.copy()
                )
                key = self.encode_condition(
                    win_r,
                    self.TURN_MAP[self.street],
                    0,
                    call_amount,
                    pot,
                    num,
                    0,
                )
                key_val = self.strat[key]
                node_t = None
                if key in self.game_state_rd:
                    node_t = self.game_state_rd[key]
                else:
                    node_t = Node(key=key, strat=list(self.strat[key]).copy())
                    self.game_state_rd[key] = node_t
                strategy = node_t.getStrategy(prob1)
                actions = [Game.FOLD, Game.CALL, Game.RAISE]
                amts = [0, call_amount, self.ante]
                util = [0, 0, 0]
                node_util = 0
                for i in range(3):
                    dup_hist = history.copy()
                    dup_hist.append((actions[i], amts[i]))
                    params_list = [
                        dup_hist,
                        p1_hole,
                        p2_hole,
                        community_cards,
                        pot,
                        prob1 * key_val[i],
                        prob2,
                        bet1 + amts[i],
                        bet2,
                        0,
                    ]
                    util[i] = self.cfr(params_list)
                    node_util += util[i] * strategy[i]
                for i in range(3):
                    node_t.regret_sum[i] += util[i] - node_util
                return node_util

            elif history[-1][0] == Game.FOLD:
                fuck = 0
                for act in history:
                    if act[0] == Game.RAISE:
                        fuck += act[1]
                return fuck
            elif history[-1][0] == Game.CALL or history[-1][0] == Game.RAISE:
                if len(history) == 1:
                    # determine the winner
                    p1_val = HandEvaluator().eval_hand(p1_hole, community_cards)
                    p2_val = HandEvaluator().eval_hand(p2_hole, community_cards)
                    pot = bet1 + bet2
                    if p1_val == p2_val:
                        return (pot / 2) - bet1
                    elif p1_val > p2_val:
                        return bet2
                    else:
                        return -bet1
                else:
                    lend =  2 + self.street if self.street >= 1 else 0
                    win_r = self.win_rate(
                        p1_hole, community_cards[0:lend], self.all_holes.copy()
                    )
                    key = self.encode_condition(
                        win_r,
                        self.TURN_MAP[len(history) // 2],
                        0,
                        call_amount,
                        pot,
                        num,
                        0,
                    )
                    key_val = self.strat[key]
                    node_t = None
                    if key in self.game_state_rd:
                        node_t = self.game_state_rd[key]
                    else:
                        node_t = Node(key=key, strat=list(self.strat[key]).copy())
                        self.game_state_rd[key] = node_t
                    strategy = node_t.getStrategy(prob1)
                    actions = [Game.FOLD, Game.CALL, Game.RAISE]
                    amts = [0, call_amount, self.ante]
                    for i in range(3):
                        dup_hist = history.copy()
                        dup_hist.append((actions[i], amts[i]))
                        params_list = [
                            dup_hist,
                            p1_hole,
                            p2_hole,
                            community_cards,
                            pot,
                            prob1,
                            prob2 * key_val[i],
                            bet1,
                            bet2 + amts[i],
                            0,
                        ]

    def get_whole_strategy(self):
        for key, node in self.game_state_rd.items():
            self.strat[key] = tuple(node.strat)
        return self.strat

class Node:
    def __init__(self, key, strat) -> None:
        self.key = key
        self.regret_sum = [0, 0, 0]
        self.strat = strat
        self.strat_sum = [0, 0, 0]

    def getStrategy(self, weight):
        norm_sum = 0
        for i in range(3):
            self.strat[i] = (
                self.regret_sum[i] if self.regret_sum[i] > 0 else max(0, self.strat[i])
            )
            norm_sum += self.strat[i]
        for i in range(3):
            if norm_sum > 0:
                self.strat[i] /= norm_sum
            else:
                self.strat[i] = 1 / 3
            self.strat_sum[i] += weight * self.strat[i]
        return self.strat


# main
strat_root = "./strat/"
start_path = "train0_strat.json"
my_cfr = CFR()
my_cfr.load_strat(strat_root + start_path)
util = my_cfr.train(50000, 1, 0)
strategy = my_cfr.get_whole_strategy()


save_path = start_path
with open(strat_root + save_path, "w") as f:
    json.dump(strategy, f, indent=4)