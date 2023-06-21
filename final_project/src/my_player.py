from game.players import BasePokerPlayer
from my_utils.card import Card
from my_utils.deck import Deck
import numpy as np
import random as rand
from my_utils.hand_evaluator import HandEvaluator
from my_utils.card_funcs import gen_deck, get_all_combinations, get_all_possible_cards
import json


class MyPlayer(BasePokerPlayer):
    def __init__(self):
        self.fold_ratio = self.call_ratio = raise_ratio = 1.0 / 3
        self.evaluator = HandEvaluator()
        self.all_cards = get_all_possible_cards()
        self.all_holes = get_all_combinations(self.all_cards, 2)
        self.ante = 50
        self.totalcost = 0
        self.sig_digits = 100
        self.pos = 0
        self.path = "./strat/train0_strat.json"
        self.strategy = None
        with open(self.path, "r") as f:
            self.strategy = json.load(f)

    ACTION_MAP_INT = {0: "fold", 1: "call", 2: "raise"}
    ACTION_MAP_STR = {v: k for k, v in ACTION_MAP_INT.items()}

    # get a hole card grades
    def hand_eval(self, hole_card, community_cards, all_hole):
        hole_hand = list(map(Card.from_str, hole_card))
        community_hand = list(map(Card.from_str, community_cards))
        return HandEvaluator().eval_hand(hole_hand, community_hand)

    def win_rate(self, hole_card, community_cards, all_hole):
        win = lose = tie = 0
        rest = 5 - len(community_cards)
        dup = self.all_cards.copy()
        for card in hole_card:
            dup.remove(card)
        for card in community_cards:
            dup.remove(card)
        rd_com = rand.choices(dup, k=rest)
        my_val = self.hand_eval(hole_card, community_cards + rd_com, all_hole)
        # print("fuck rd_com\n\n", rd_com," \n\n")
        for hole in all_hole:
            if hole[0] in hole_card + community_cards + rd_com:
                continue
            if hole[1] in hole_card + community_cards + rd_com:
                continue
            opp_val = self.hand_eval(hole, community_cards + rd_com, all_hole)
            if my_val > opp_val:
                win += 1
            elif my_val == opp_val:
                tie += 1
            else:
                lose += 1
        return (win + (tie) / 2) / (win + tie + lose)

    def eval_action(self, valid_action, win_rate, round_state):
        pot = round_state["pot"]["main"]["amount"]
        # fold
        fold_exp = self.totalcost * (-1)
        # call
        after_me = self.n_player - self.pos - 1
        call_exp = (pot + after_me * valid_action[1]["amount"]) * win_rate
        call_exp -= (1 - win_rate) * (self.totalcost + valid_action[1]["amount"])
        # raise
        raise_amount = valid_action[2]["amount"]["min"] + 1
        raise_exp = (pot + after_me * raise_amount) * win_rate
        raise_exp -= (1 - win_rate) * (self.totalcost + raise_amount)
        return [fold_exp, call_exp, raise_exp]

    def naive_decider(self, valid_actions, hole_cards, community_cards):
        win_r = self.win_rate(hole_cards, community_cards, self.all_holes)
        action = "call"
        info = valid_actions[1]
        amount = info["amount"]
        print("fuck win rate\n\n", win_r, "\n\n")
        [fold_exp, call_exp, raise_exp] = self.eval_action(
            valid_actions, win_r, self.round_state
        )
        print("fuck exp\n\n", fold_exp, call_exp, raise_exp, "\n\n")
        if call_exp > 0:
            opt = (self.pos + 1) / self.n_player
            if raise_exp > call_exp and opt < 0.6:
                action = "raise"
                amount = valid_actions[2]["amount"]["min"] + 1
                amount *= 10 * win_r + 1
                amount = min(amount, valid_actions[2]["amount"]["max"])
            else:
                action = "call"
                amount = valid_actions[1]["amount"]
        else:
            action = "fold"
            amount = 0
        return action, amount

    def encode_condition(self, win_rate, round_state, valid_actions):
        street = round_state["street"]
        position = self.pos
        stt = 0
        if street == "preflop":
            stt = 0
        elif street == "flop":
            stt = 1
        elif street == "turn":
            stt = 2
        elif street == "river":
            stt = 3
        pos = position + 1 / self.n_player
        if pos > 0.5:
            pos = 1
        else:
            pos = 0
        rt = int(win_rate * self.sig_digits)
        if rt >= self.sig_digits:
            rt = self.sig_digits - 1
        rt %= self.sig_digits
        pos = 0
        ret = "%d_%d_%d" % (stt, rt, pos)
        return ret

    def strategy_decider(self, valid_actions, hole_cards, round_state):
        community_cards = round_state["community_card"]
        win_r = self.win_rate(hole_cards, community_cards, self.all_holes)
        print("win rate\n\n", win_r, "\n\n")
        action = "call"
        info = valid_actions[1]
        amount = info["amount"]
        key = self.encode_condition(win_r, round_state, valid_actions)
        #print("fuck key\n\n", key, "\n\n")
        act_list = list(self.strategy[key])
        #print("fuck actlist\n\n", act_list[1], "\n\n")
        action = 0
        rat = act_list[0]
        if rat < act_list[1]:
            rat = act_list[1]
            action = 1
        if rat < act_list[2]:
            rat = act_list[2]
            action = 2
        action = self.ACTION_MAP_INT[action]
        print("fuck action strat\n\n", action, "\n\n")
        if action == "raise":
            constant = min(act_list[2] * 10 - 1, 5)
            amount = (
                valid_actions[2]["amount"]["min"] * (1 + win_r)*constant + 1
            )
            amount = min(amount, valid_actions[2]["amount"]["max"])
        elif action == "call":
            amount = info["amount"]
        elif action == "fold":
            amount = 0
        return action, amount

    def declare_action(self, valid_actions, hole_card, round_state):
        # print("fuck round_state\n\n", round_state["small_blind_pos"], "\n\n")
        self.round_state = round_state
        action = "call"
        amount = valid_actions[1]["amount"]
        if round_state["street"] == "preflop":
            pass
        else:
            action, amount = self.strategy_decider(
                valid_actions, hole_card, round_state
            )
            # action, amount = self.naive_decider(
            # valid_actions, hole_card, round_state["community_card"]
            # )
        print("fuck action\n\n", action, "  ", amount, "\n\n")
        self.totalcost += amount
        return action, amount

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.totalcost = 0
        self.n_player = len(seats)
        pass

    def receive_street_start_message(self, street, round_state):
        seats = round_state["seats"]
        for i in range(len(seats)):
            if seats[i]["uuid"] == self.uuid:
                self.pos = i
                self.name = seats[i]["name"]
        sb_p = round_state["small_blind_pos"]
        if self.pos >= sb_p:
            self.pos -= sb_p
        else:
            self.pos = len(seats) - sb_p + self.pos
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return MyPlayer()
