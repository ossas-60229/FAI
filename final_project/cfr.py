import random
import json
from my_utils.card_funcs import gen_deck, get_all_combinations, get_all_possible_cards
from my_utils.card import Card
from my_utils.hand_evaluator import HandEvaluator
from my_utils.Mygame import Game
import numpy as np


class My_CFR:
    def __init__(self) -> None:
        self.game_state_rd = dict()
