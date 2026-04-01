import torch
from enum import Enum

from omados.engine.cards import OBER, UNTER, ASSE, SUIT_MASKS, Suit, Cards, INDEX_TO_SUIT

class Spielart(Enum): 
    Sauspiel = 0
    Wenz = 1 
    Geier = 2 
    Ramsch = 3 
    
    def get_trumpf_cards(self, suit: Suit) -> Cards:
        suit_cards = SUIT_MASKS[suit] if suit is not None else Cards(torch.zeros(32, dtype=torch.bool))
        if self == Spielart.Sauspiel or self == Spielart.Ramsch:
            return OBER | UNTER | suit_cards
        elif self == Spielart.Wenz:
            return UNTER | suit_cards
        elif self == Spielart.Geier:
            return OBER | suit_cards
        else: 
            raise ValueError(f"Invalid Spielart: {self}")


def get_cards() -> Cards: 
    card_indices = torch.randperm(32)
    player_cards = torch.zeros((4, 32), dtype=torch.bool)
    for i in range(4):
        player_cards[i, card_indices[i*8:(i+1)*8]] = True
    return [Cards(player_cards[i]) for i in range(4)]

def is_trumpf(card: int, trumpf_cards: Cards): 
    return trumpf_cards[card]

def get_playable_cards(player_cards: Cards, gesuchte_farbe: Suit, trumpf_cards: Cards, angespielte_karte: int = -1): 
    # if angespielte_karte is not -1, player is not first
    if angespielte_karte != -1: 
        if is_trumpf(angespielte_karte, trumpf_cards): 
            same_color_cards : Cards = player_cards & trumpf_cards
        else:
            angespielte_farbe : Suit = INDEX_TO_SUIT[angespielte_karte]
            same_color_cards : Cards = player_cards & SUIT_MASKS[angespielte_farbe] & ~trumpf_cards
        
        # if kann zugeben
        if same_color_cards.any():
            playable_cards : Cards = same_color_cards
        else: 
            playable_cards = player_cards    
    else: 
        playable_cards = player_cards
         
    # if player has ass of the gesuchte_farbe, they must play it if its color is angespielt, except if they have at least four cards
    gesuchte_farbe_ass : Cards = player_cards & ASSE & SUIT_MASKS[gesuchte_farbe]
    if gesuchte_farbe_ass.any(): 
        gesuchte_farbe_cards : Cards = player_cards & SUIT_MASKS[gesuchte_farbe]
        can_run_away : bool = (gesuchte_farbe_cards.sum() >= 4) and (angespielte_karte == -1)
        
        if not can_run_away:
            temp : Cards = ~SUIT_MASKS[gesuchte_farbe] | gesuchte_farbe_ass 
            playable_cards &= temp
    
    return playable_cards

def get_playable_suits(player_cards: Cards, trumpf_cards: Cards): 
    # one can only play if they have the color they want to play and not the corresponding ace
    non_trumpf_cards : Cards = player_cards & ~trumpf_cards
    
    playable_suits = []
    
    for suit in Suit:
        suit_cards = non_trumpf_cards & SUIT_MASKS[suit]
        if (not (suit_cards & ASSE).any()) and (suit_cards & ~ASSE).any(): 
            playable_suits.append(suit)
    
    return playable_suits    
    
    # ass_farben_auf_hand = torch.zeros(4, dtype=torch.bool)
    # ass_farben_auf_hand[farb_indices[farb_cards & assen].unique()] = True
    
    # farb_karten_ohne_ass = torch.zeros(4, dtype=torch.bool)
    # farb_karten_ohne_ass[farb_indices[farb_cards & ~assen].unique()] = True