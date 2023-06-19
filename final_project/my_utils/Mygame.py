import random
from my_utils.card_funcs import gen_deck
from my_utils.card import Card
import numpy as np

class Game:
    def __init__(self) -> None:
        pass
    RAISE = "0"
    CALL = "1"
    FOLD = "2"
    SMALL_BLIND = "3"
    BIG_BLIND = "4"
    DECK = gen_deck()
    def deal_cards():
        sample = random.sample(Game.DECK.deck, 4)
        player_one_cards = sample[0:2]
        player_two_cards = sample[2:4]
        return player_one_cards, player_two_cards

    @staticmethod
    def get_higher_rank(card1, card2):
        if card1.rank > card2.rank:
            return card1
        return card2
    
    @staticmethod
    def get_higher_suit(card1, card2):
        if card1.suit > card2.suit:
            return card1
        elif card1.suit == card2.suit:
            return 0
        return card2
