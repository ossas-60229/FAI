
from game.players import BasePokerPlayer
import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np
import random as rand

model_root = "./models/"
model_path = [None, None, None, None]
model_path[0] = "./models/model_pre.pt"
model_path[1] = "./models/model_flop.pt"
model_path[2] = "./models/model_turn.pt"
model_path[3] = "./models/model_river.pt"
model_raise = "./models/model_raise.pt"

class MYCARD:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return str(self.suit) + str(self.rank)

    def __repr__(self):
        return "[" + str(self.suit) + ", " + str(self.rank) + "]"

class MyPlayer_naive(
    BasePokerPlayer
):  # Do not forget to make parent class as "BasePokerPlayer"
    round_map = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
    rank_map = {
        "A": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "T": 10,
        "J": 11,
        "Q": 12,
        "K": 13,
    }
    suit_map = {"C": 0, "D": 1, "H": 2, "S": 3}
    operation_map = {"RAISE": 0, "CALL": 1, "FOLD": 2, "SMALLBLIND": 3, "BIGBLIND": 4}

    # playing
    hand_list = {
        "Royal_flush": 0,
        "Straight_flush": 1,
        "Four_of_a_kind": 2,
        "Full_house": 3,
        "Flush": 4,
        "Straight": 5,
        "Three_of_a_kind": 6,
        "Two_pair": 7,
        "One_pair": 8,
        "High_card": 9,
    }
    # hand and its code
    hand_list_prob = {
        "Royal_flush": 0.00000039,
        "Straight_flush": 0.00001539,
        "Four_of_a_kind": 0.0002401,
        "Full_house": 0.00144058,
        "Flush": 0.0019654,
        "Straight": 0.00392465,
        "Three_of_a_kind": 0.02112845,
        "Two_pair": 0.04753902,
        "One_pair": 0.42256903,
        "High_card": 0.50117739,
    }

    # maps
    def decode_card(self, card_str) -> MYCARD:
        ret = MYCARD(self.suit_map[card_str[0]], self.rank_map[card_str[1]])
        return ret

    def __init__(self):
        self.seat = 0
        self.my_info = None
        self.players_map = dict()  # map players into seat number
        self.player_states = None
        self.game_info = None
        self.round_state = None
        self.hand_info = None
        self.winners = None
        self.hole_card = []
        self.public = []
        self.name = None
        self.whole_cards = []
        for i in range(4):
            for j in range(1, 14):
                self.whole_cards.append(MYCARD(i, j))
        # game board basic information

    def declare_action(self, valid_actions, hole_card, round_state):
        self.hole_card = []
        for fuck in hole_card:
            self.hole_card.append(self.decode_card(fuck))
        # the hole cards
        self.public = []
        for fuck in round_state["community_card"]:
            self.public.append(self.decode_card(fuck))
        action = "call"
        amount = valid_actions[1]["amount"]
        self.total_cost += amount
        return action, amount  # action returned here is sent to the poker engine

    def gameinfo_decoder(self):
        fuck = 0
        for player in self.game_info["seats"]:
            self.players_map[player["uuid"]] = fuck
            fuck += 1
            if self.uuid == player["uuid"]:
                self.my_info = player
        # initial state
        # know my name

    def receive_game_start_message(self, game_info):
        # print("fuck game_info\n\n\n\n", game_info, "\n\n\n\nfuck")
        self.sb = game_info["rule"]["small_blind_amount"]
        self.stack = game_info["rule"]["initial_stack"]
        # self.sb = 0
        self.bb = 2 * self.sb
        self.game_info = game_info
        self.gameinfo_decoder()
        return

    def receive_round_start_message(self, round_count, hole_card, seats):
        # print("fuck rdcnt\n\n\n\n", round_count, "\n\n\n\nfuck")
        self.total_cost = 0
        self.n_round = round_count
        self.hole_card = hole_card
        self.seats = seats
        return

    def receive_street_start_message(self, street, round_state):
        # print("fuck rount state\n\n\n\n", round_state, "\n\n\n\nfuck")
        # print("fuck street\n\n\n\n", street, "\n\n\n\nfuck")
        self.street = street
        self.round_state = round_state
        return
        pass

    def receive_game_update_message(self, action, round_state):
        # print("fuck action\n\n\n\n", action, "\n\n\n\nfuck")
        # print("fuck round state\n\n\n\n", round_state, "\n\n\n\nfuck")
        self.round_state = round_state
        return

    def receive_round_result_message(self, winners, hand_info, round_state):
        # print("fuckwinner\n\n\n\n", winners, "\n\n\n\nfuck")
        # print("fuckhandinfo\n\n\n\n", hand_info, "\n\n\n\nfuck")
        seat = round_state["seats"]
        stack = self.stack
        for player in seat:
            if player["uuid"] == self.uuid:
                self.stack = player["stack"]
        self.round_state = round_state
        self.hand_info = hand_info
        self.winners = winners
        self.win = 0
        return

    def eval_poker_hands(self, card_list):  # return (type, rank)
        # determine what type and rank of a poker hand(5 cards)
        def get_suit(card: MYCARD):
            return card.suit

        def get_rank(card: MYCARD):
            return card.rank

        s_suit = False
        list.sort(card_list, key=get_suit)
        if card_list[0].suit == card_list[-1].suit:
            s_suit = True
        # print("fuck list\n\n", card_list, "\n\n")
        list.sort(card_list, key=get_rank)
        # print("fuck list\n\n", card_list, "\n\n")
        ret_type = ""
        ret_rank = 0
        if s_suit:
            # Maybe straight flush or royal flush
            rlist = [card.rank for card in card_list]
            if rlist == [1, 10, 11, 12, 13]:
                ret_type = "Royal_flush"
                ret_rank = 1
            elif rlist[-1] - rlist[0] == 4:
                ret_type = "Straight_flush"
                ret_rank = rlist[-1]
            else:
                ret_type = "Flush"
                ret_rank = rlist[-1]
        else:
            same_rank = 1
            tmp_list = list()
            for i in range(4):
                if card_list[i].rank == card_list[i + 1].rank:
                    same_rank += 1
                else:
                    tmp_list.append([same_rank, card_list[i].rank])
                    same_rank = 1
            # print("fuck you\n\n\n\n\n\n")
            tmp_list.append([same_rank, card_list[4].rank])
            tmp_list.sort()
            num_list = [len(seg) for seg in tmp_list]
            if num_list == [1, 4]:
                ret_type = "Four_of_a_kind"
                ret_rank = tmp_list[-1][1]
            elif num_list == [2, 3]:
                ret_type = "Full_house"
                ret_rank = tmp_list[-1][1]
            elif num_list == [1, 1, 3]:
                ret_type = "Three_of_a_kind"
                ret_rank = tmp_list[-1][1]
            elif num_list == [1, 2, 2]:
                ret_type = "Two_pair"
                ret_rank = max(tmp_list[-1][1], tmp_list[-2][1])
            elif num_list == [1, 1, 1, 2]:
                ret_type = "One_pair"
                ret_rank = tmp_list[-1][1]
            else:
                if card_list[-1].rank - card_list[0].rank == 4 or (
                    card_list[-1].rank - card_list[1].rank == 3
                    and card_list[0].rank == 1
                ):
                    ret_type = "Straight"
                    ret_rank = card_list[-1].rank
                else:
                    ret_type = "High_card"
                    ret_rank = card_list[4].rank
        return ret_type, ret_rank

    def grade_hands(self, type, rank):
        tpe = 9 - self.hand_list[type]
        return tpe * 13 + rank

    def best_in_hand(self):  # return (type rank)
        # fuck_list = [
        # MYCARD(0, 10),
        # MYCARD(0, 11),
        # MYCARD(0, 12),
        # MYCARD(0, 13),
        # MYCARD(0, 9),
        # ]
        # print(self.eval_poker_hands(fuck_list), "\n\n\n\nn\n")
        best_type = ("High_card", 0)
        l = len(self.public)
        card_list = [card for card in self.hole_card]
        for i in range(0, l - 2):
            card_list.append(self.public[i])
            for j in range(i + 1, l - 1):
                card_list.append(self.public[j])
                for k in range(j + 1, l):
                    card_list.append(self.public[k])
                    type, rank = self.eval_poker_hands(card_list.copy())
                    if self.hand_list[type] < self.hand_list[best_type[0]]:
                        best_type = (type, rank)
                    if self.hand_list[type] == self.hand_list[best_type[0]]:
                        if rank > best_type[1]:
                            best_type = (type, rank)
                    card_list.pop()
                card_list.pop()
            card_list.pop()
        # use two hole cards
        card_list.clear()
        if l > 3:
            for i in range(l):
                card_list = [card for card in self.public]
                card_list.remove(self.public[i])
                for i in len(self.hole_card):
                    card_list.append(self.hole_card[i])
                    type, rank = self.eval_poker_hands(card_list.copy())
                    if self.hand_list[type] < self.hand_list[best_type[0]]:
                        best_type = (type, rank)
                    if self.hand_list[type] == self.hand_list[best_type[0]]:
                        if rank > best_type[1]:
                            best_type = (type, rank)
                    card_list.pop()
        # use only one hole card
        card_list.clear()
        if l > 4:
            card_list = [card for card in self.public]
            type, rank = self.eval_poker_hands(card_list.copy())
            if self.hand_list[type] < self.hand_list[best_type[0]]:
                best_type = (type, rank)
            if self.hand_list[type] == self.hand_list[best_type[0]]:
                if rank > best_type[1]:
                    best_type = (type, rank)
        # use no hole cards (every one the same !)

        return best_type

    def min_max_commu(self, c_cards):
        def get_suit(card: MYCARD):
            return card.suit

        def get_rank(card: MYCARD):
            return card.rank

        l = len(c_cards)
        ret_type = "High_card"
        ret_rank = 0
        if l == 3:
            list.sort(c_cards, key=get_rank)
            if c_cards[0].rank == c_cards[-1].rank:
                ret_type = "Three_of_a_kind"
                ret_rank = c_cards[0].rank
            else:
                same_rank = 0
                fuck = 0
                for i in range(0, 2):
                    for j in range(i + 1, 3):
                        if c_cards[i].rank == c_cards[j].rank:
                            same_rank += 1
                            fuck = max(fuck, c_cards[i].rank)
                if same_rank > 0:
                    ret_type = "One_pair"
                    ret_rank = c_cards[0].rank
        elif l == 4:
            list.sort(c_cards, key=get_rank)
            if c_cards[0].rank == c_cards[-1].rank:
                ret_type = "Four_of_a_kind"
                ret_rank = c_cards[0].rank
            elif c_cards[0].rank == c_cards[-2].rank:
                ret_type = "Three_of_a_kind"
                ret_rank = c_cards[0].rank
            elif c_cards[1].rank == c_cards[-1].rank:
                ret_type = "Three_of_a_kind"
                ret_rank = c_cards[1].rank
            else:
                same_rank = 0
                fuck = 0
                for i in range(0, 3):
                    for j in range(i + 1, 4):
                        if c_cards[i].rank == c_cards[j].rank:
                            same_rank += 1
                            fuck = max(fuck, c_cards[i].rank)
                if same_rank == 1:
                    ret_type = "One_pair"
                    ret_rank = c_cards[0].rank
                elif same_rank == 2:
                    ret_type = "Two_pair"
                    ret_rank = c_cards[0].rank
                else:
                    ret_type = "High_card"
                    ret_rank = c_cards[0].rank

        elif l == 5:
            ret_type, ret_rank = self.eval_poker_hands(c_cards.copy())
        return ret_type, ret_rank

    def max_max_commu(self, c_cards):
        def get_suit(card: MYCARD):
            return card.suit

        def get_rank(card: MYCARD):
            return card.rank

        l = len(c_cards)
        ret_type = "High_card"
        ret_rank = 0
        list.sort(c_cards, key=get_suit)
        c_bysuit = [[], [], [], []]
        for card in c_cards:
            c_bysuit[card.suit].append(card)
        royal_sequenc = [10, 11, 12, 13, 1]
        ret_type = "Straight_flush"
        return ret_type, 13

    def analyze_community_cards(self, c_cards):
        # analyze the probability of community cards
        def get_suit(card: MYCARD):
            return card.suit

        def get_rank(card: MYCARD):
            return card.rank

        l = len(c_cards)
        prob_list = [0 for _ in range(len(self.hand_list))]
        rank_list = [0 for _ in range(len(self.hand_list))]
        # probability of royal_flush
        section_type = "Royal_flush"
        royal_sequenc = [10, 11, 12, 13, 1]
        match_list = [0, 0, 0, 0]
        for crd in c_cards:
            if crd.rank in royal_sequenc:
                match_list[crd.suit] += 1
        max_match = max(match_list)
        prob_list[self.hand_list[section_type]] = max_match
        # probability of straight_flush
        section_type = "Straight_flush"
        list.sort(c_cards, key=get_rank)
        match_list = [[], [], [], []]
        for crd in c_cards:
            match_list[crd.suit].append(crd)
        max_match = 0
        for clist in match_list:
            if len(clist) < 2:
                max_match = max(max_match, len(clist))
            else:
                list.sort(clist, key=get_rank)
                l2 = len(clist)
                for i in range(l2):
                    if clist[l - i - 1].rank - clist[0].rank <= 4:
                        max_match = max(max_match, l - i)

        prob_list[self.hand_list[section_type]] = (1 / 50) ** (5 - max_match)

        # probability of four of a kind
        section_type = "Four_of_a_kind"
        mth = 1
        max_match = 0
        rank = c_cards[0].rank
        for i in range(l - 1):
            if get_rank(c_cards[i]) == get_rank(c_cards[i + 1]):
                mth += 1
            else:
                mth = 1
            max_match = max(max_match, mth)
        prob_list[self.hand_list[section_type]] = (1 / 50) ** (5 - max_match)

        # probability of full house
        section_type = "Full_house"
        max_match = min(max_match, 3)
        prob_list[self.hand_list[section_type]] = (1 / 50) ** (5 - max_match)

        # probability of flush
        list.sort(c_cards, key=get_suit)
        mth = 1
        max_match = 0
        for i in range(l - 1):
            if get_suit(c_cards[i]) == get_suit(c_cards[i + 1]):
                mth += 1
            else:
                mth = 1
            max_match = max(max_match, mth)
        prob_list[self.hand_list[section_type]] = (1 / 50) ** (5 - max_match)

        # probability of straight
        list.sort(c_cards, key=get_rank)
        mth = 1
        max_match = 0
        for i in range(0, l - 1):
            for j in range(i, l):
                mth = get_rank(c_cards[j]) - get_rank(c_cards[i])
                if mth <= 4:
                    max_match = max(max_match, j - i)
        prob_list[self.hand_list[section_type]] = (4 / 50) ** (5 - max_match)

        # probability of three of a kind
        section_type = "Three_of_a_kind"
        mth = 1
        max_match = 0
        rank = c_cards[0].rank
        for i in range(l - 1):
            if get_rank(c_cards[i]) == get_rank(c_cards[i + 1]):
                mth += 1
            else:
                mth = 1
            max_match = max(max_match, mth)
        prob_list[self.hand_list[section_type]] = (1 / 50) ** min((3 - max_match), 0)

        # probability of two pair
        section_type = "Two_pair"
        mth = 1
        max_match = 0
        fk = 0
        rank = c_cards[0].rank
        for i in range(l - 1):
            if get_rank(c_cards[i]) == get_rank(c_cards[i + 1]):
                mth += 1
            else:
                mth = 1
            if mth > 1:
                fk += 1
            max_match = max(max_match, mth)
        prob_list[self.hand_list[section_type]] = (1 / 50) ** min((2 - fk) * 2, 0)
        # probability of one pair
        section_type = "One_pair"
        prob_list[self.hand_list[section_type]] = (1 / 50) ** min((2 - max_match), 0)

        # probability of high card
        prob_list[self.hand_list["High_card"]] = 1
        return prob_list

    def eval_cmt_card(self, c_cards):
        return self.analyze_community_cards(c_cards.copy())
        l = len(c_cards)
        if l == 3:
            return 0
        elif l == 4:
            return 1
        elif l == 5:
            return 2

    def decision_maker_naive(self, valid_actions, hole_card, round_state):
        move = "call"
        state = ""
        info = valid_actions[1]
        street = round_state["street"]
        btype, rank = self.best_in_hand()
        amount = info["amount"]
        if self.round_map[street] == 0:
            # preflop
            state = "preflop"
            move = "call"
            amount = info["amount"]
        elif self.round_map[street] >= 1:
            # flop
            state = "flop"
            min_max_type, min_max_rank = self.min_max_commu(self.public.copy())
            max_max_type, max_max_rank = self.max_max_commu(self.public.copy())
            max_val = self.grade_hands(max_max_type, max_max_rank)
            min_val = self.grade_hands(min_max_type, min_max_rank)
            my_val = self.grade_hands(btype, rank)
            heur = (my_val - min_val) / (max_val - min_val)
            # print("fuck heur\n\n\n\n", heur, "\n\n\n\nfuck")
            if heur > 10:
                move = "raise"
                amount = valid_actions[2]["amount"]["min"] * (1 + heur)
                # print("fuck amount\n\n\n\n", amount, "\n\n\n\nfuck")
            else:
                move = "call"
                amount = valid_actions[1]["amount"]
        # print("fuck your move\n\n\n\n", move, "\n\n\n\nfuck")
        # elif self.round_map[street] == 2:
        ## turn
        # state = "turn"
        # elif self.round_map[street] == 3:
        ## river
        # state = "river"
        # information
        return move, amount

