
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

class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.i_model = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim),
            nn.Softmax(dim=1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.005)
        self.loss_func = nn.MSELoss()
        # self.i_model = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.i_model(x)
        return x

    def fuck(self):
        # print("fuck\n\n\n")
        return

    def small_fit(self, x, y):
        tmp = self.loss_func(self.forward(x), y)
        self.optimizer.zero_grad()
        tmp.backward()
        self.optimizer.step()


class MyPlayer_model(
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
        self.model = [None for i in range(4)]
        self.output = [None for i in range(4)]
        self.input = [None for i in range(4)]
        self.output_bet = [None for i in range(4)]
        self.input_bet = [None for i in range(4)]
        self.n_features = 7
        self.n_operations = 3
        self.bet_model = None
        for i in range(4):
            self.load_model(i)

    def init_bet_model(self):
        # input: pure winrate,  action and states of opponents, pot
        # output: (possible benefit rate, possible loss rate)
        # predict the possible rate of benefit
        seat = self.round_state["seats"]
        i_d = 1 + len(seat) * 2 + 1
        o_d = 2
        self.bet_model = MyModel(i_d, o_d)
        return None

    # training
    def create_model(self, i):
        # predict a win rate``
        if i == 0:
            self.n_features = 2 + 2
            self.model[i] = MyModel(self.n_features, 2)
        else:
            # flop
            in_h = 2
            in_c = 3 + i - 1
            num = in_h * (in_h - 1) / 2 + in_c * (in_c - 1) / 2 + in_h * in_c
            num *= 2
            num += in_h + in_c
            num = int(num)
            # print("num = ", num, "\n\n\n\n")
            # rank distance and suit similarity of each pair
            self.model[i] = MyModel(num, 2)
        torch.save(self.model[i], model_path[i])

    def load_model(self, i):
        fp = None
        creat_model = False
        if os.path.exists(model_path[i]):
            fp = open(model_path[i], "rb")
            if os.path.getsize(model_path[i]) == 0:
                creat_model = True
            else:
                creat_model = False
        else:
            if not os.path.exists(model_root):
                os.mkdir(model_root)
            fp = open(model_path[i], "a")
            creat_model = True
        fp.close()
        if creat_model:
            # create a new model
            self.create_model(i)
        else:
            self.model[i] = torch.load(model_path[i])
        return

    def preflop_encoding(self, hole_cards):
        def get_rank(card: MYCARD) -> int:
            return card.rank

        def get_suit(card: MYCARD) -> int:
            return card.suit

        ret_arr = np.zeros((1, 4))
        tmp_c = hole_cards[:]
        list.sort(tmp_c, key=get_suit)
        if tmp_c[0].suit == tmp_c[1].suit:
            ret_arr[0, 0] = 1
        ret_arr[0, 1] = tmp_c[1].rank - tmp_c[0].rank
        ret_arr[0, 2] = tmp_c[0].rank
        ret_arr[0, 3] = tmp_c[1].rank
        ret = torch.Tensor(ret_arr)
        # print("encoded\n\n")
        return ret

    def preflop_forward_simple(self, input):
        # upper TENSOR ==
        self.input[0] = input
        ret = self.model[0](input)
        self.output[0] = ret
        # print("model is ", ret)
        # print("forward end")
        return ret

    def other_encoding(self, hole_cards, community_cards):
        # except for preflop
        l1 = len(hole_cards)
        l2 = len(community_cards)
        ret_list = []
        # between 2 whole cards
        for i in range(l1 - 1):
            for j in range(i + 1, l1):
                ret_list.append(abs(hole_cards[i].rank - hole_cards[j].rank))
                # distance
                bool = 0
                if hole_cards[i].suit == hole_cards[j].suit:
                    bool = 1
                ret_list.append(bool)
                # similarity
        # between 2 community cards
        for i in range(l2 - 1):
            for j in range(i + 1, l2):
                ret_list.append(abs(community_cards[i].rank - community_cards[j].rank))
                # distance
                bool = 0
                if community_cards[i].suit == community_cards[j].suit:
                    bool = 1
                ret_list.append(bool)
                # similarity
        # between hole cards and community cards
        for i in range(l1):
            for j in range(l2):
                ret_list.append(abs(hole_cards[i].rank - community_cards[j].rank))
                # distance
                bool = 0
                if hole_cards[i].suit == community_cards[j].suit:
                    bool = 1
                ret_list.append(bool)
                # similarity
        for i in range(l1):
            ret_list.append(hole_cards[i].rank)
        for i in range(l2):
            ret_list.append(community_cards[i].rank)
        return torch.Tensor([ret_list])

    def other_forward_simple(self, input, i):
        input = torch.Tensor(input)
        # print("fuck input[i] = \n\n\n\n", input, "\n\n\n\n")
        # print("fuck model[i] = \n\n\n\n", self.model[i], "\n\n\n\n")
        self.input[i] = input
        ret = self.model[i](input)
        return ret

    def fit_model_simple(self, i, win_or_lose):
        # print("fuck input[i] = \n\n\n\n", self.input[i], "\n\n\n\n")
        self.model[i].small_fit(
            self.input[i], torch.Tensor([[win_or_lose, 1 - win_or_lose]])
        )
        return

    def bet_encoding(self, winrate, round_state):
        ret_arr = []
        ret_arr.append(winrate)
        seats = round_state["seats"]
        l = len(seats)
        for i in range(l):
            player = seats[(i + round_state["small_blind_pos"]) % l]
            ret_arr.append(player["stack"])
            tmp = 0
            if player["state"] == "participating":
                tmp = 1
            ret_arr.append(tmp)
        record = round_state["action_histories"][round_state["street"]]
        # print("fuck round state\n\n\n\n", round_state, "\n\n\n\n")
        # for rd in record:
        # if rd["action"] == "FOLD":
        # ret_arr.append(0)
        # else:
        # ret_arr.append(rd["amount"])
        ret_arr.append(round_state["pot"]["main"]["amount"])
        return torch.Tensor([ret_arr])

    def bet_forward(self, input):
        self.street = self.round_state["street"]
        i = self.round_map[self.street]
        self.input_bet[i] = input
        ret = self.bet_model(input)
        return ret

    def bet_fit_model(self, stack, new_stack, i):
        loss = new_stack - stack
        input_result = torch.Tensor([[loss / stack, 1 - loss / stack]])
        # print("input_bet[i] = \n\n\n\n", self.input_bet[i], "\n\n\n\n")
        # print("model is \n\n\n\n", self.bet_model, "\n\n\n\n")
        self.bet_model.small_fit(self.input_bet[i], input_result)

    def sim_winrate_predict(self, hole_cards, community_cards, street):
        output = 0
        if street == "preflop":
            input = self.preflop_encoding(hole_cards)
            output = self.preflop_forward_simple(input)
        else:
            i = self.round_map[street]
            input = self.other_encoding(hole_cards, community_cards)
            output = self.other_forward_simple(input, i)
        winrate = output[0][0].item()
        return winrate

    def sim_benefit_predict(self, win_rate, round_state):
        input = self.bet_encoding(win_rate, round_state)
        ret = self.bet_forward(input)
        return ret[0][0].item()

    # probabilities (from Wikipedia)
    def sim_decider(self, valid_actions, hole_card, community_card, round_state):
        ret_action = "call"
        ret_amount = valid_actions[1]["amount"]
        winrate = self.sim_winrate_predict(
            hole_card, community_card, round_state["street"]
        )
        be_rate = self.sim_benefit_predict(winrate, round_state)
        while be_rate < 0.001:
            be_rate *= 10
        print("\n\nfuck win rate = ", winrate, "\n\n")
        exp_fold = (-1) * self.total_cost
        pot = round_state["pot"]["main"]["amount"]
        info = valid_actions[1]
        street = self.round_map[round_state["street"]]
        street = 3 - street
        num = len(round_state["seats"])
        exp_call = winrate * (pot + info["amount"] * street * num) - (1 - winrate) * (
            self.total_cost + info["amount"] * street
        )
        if exp_call > exp_fold:
            ret_action = "call"
            ret_amount = valid_actions[1]["amount"]
        else:
            ret_action = "fold"
            ret_amount = 0

        return ret_action, ret_amount

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
        action, amount = self.sim_decider(
            valid_actions, self.hole_card.copy(), self.public.copy(), round_state
        )
        print("\n\naction, amount: ", action, amount, "\n\n")
        # print("fuckfucckadas action\n\n\n\n", action, amount, "\n\n\n\nfuck")
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
        if os.path.exists(model_raise):
            self.bet_model = torch.load(model_raise)
        else:
            self.init_bet_model()
            torch.save(self.bet_model, model_raise)
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
        for winner in winners:
            if winner["uuid"] == self.uuid:
                self.win = 1
                print(" I win!!!\n")
        for i in range(4):
            if self.input[i] != None:
                self.fit_model_simple(i, self.win)
                torch.save(self.model[i], model_path[i])
        for i in range(4):
            if self.input_bet[i] != None:
                self.bet_fit_model(stack, self.stack, i)
                torch.save(self.bet_model, model_raise)
        return
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

