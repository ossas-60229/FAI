import random as rd
import numpy as np
from my_utils.card import Card
from my_utils.hand_evaluator import HandEvaluator
from my_utils.deck import Deck
from itertools import combinations


def gen_cards(cards_str):
    #generate cards
    return [Card.from_str(s) for s in cards_str]

def gen_deck(exclude_cards=None):
    deck_ids = range(1, 53)
    if exclude_cards:
        assert isinstance(exclude_cards, list)
        if isinstance(exclude_cards[0], str):
            exclude_cards = [Card.from_str(s) for s in exclude_cards]
        exclude_ids = [card.to_id() for card in exclude_cards]
        deck_ids = [i for i in deck_ids if not i in exclude_ids]
    return Deck(deck_ids)
def set_product(set1, set2):
    res = []
    for i in set1:
        for j in set2:
            res.append(i+j)
    return res


def get_all_possible_cards():
    numbers = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
    suites = ["S", "D", "H", "C"]
    return set_product(suites, numbers)


def get_all_combinations(cards, n):
    return list(combinations(cards, n))

def evaluate_all_flops():
    all_cards = get_all_possible_cards()
