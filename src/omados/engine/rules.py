import torch
from .cards import Cards, SUIT_MASKS, ASSE, SUIT_LOOKUP, CARD_POINTS, INDEX_TO_SUIT, Suit
from .modes import GameContract
from .tricks import Trick

def get_legal_moves(
    player_hand: Cards, 
    lead_card_idx: int, 
    contract: GameContract,
    ran_away: bool = False
) -> Cards:
    
    """Returns a mask of playable cards based on Schafkopf rules."""
    trumpf = contract.trumpf_cards
    
    # 1. Lead Player (First in trick)
    if lead_card_idx == -1:
        # Standard Schafkopf: If you have the 'Rufsau' (Ace), 
        # you cannot lead the Rufsau suit with a non-ace unless you have 4+ of them.
        if (contract.rufsau & player_hand).any() and not ran_away:
        # if (contract.rufsau & player_hand).any():
            rufsau_suit_cards = player_hand & SUIT_MASKS[contract.rufsau_suit]
            if rufsau_suit_cards.sum() < 4:
                return (player_hand & ~SUIT_MASKS[contract.rufsau_suit]) | contract.rufsau
        return player_hand

    # 2. Follower Logic
    is_lead_trumpf = trumpf[lead_card_idx]
    
    # playable = cards that follow the lead card (i.e. Zugeben)
    if is_lead_trumpf:
        playable = player_hand & trumpf
    else:
        lead_suit = INDEX_TO_SUIT[lead_card_idx]
         # Must play same suit, but only the 'Farbe' part (not trumps)
        playable = player_hand & SUIT_MASKS[lead_suit] & ~trumpf
    
    # if player did not ran away and the lead suit (non-trumpf) is of the rufsau suit, they can only play that.
    if not ran_away: 
        if playable.any(): 
            legal_moves = playable & (contract.rufsau | ~SUIT_MASKS[contract.rufsau_suit])
        else:
            legal_moves = player_hand & ~contract.rufsau
    else:
        if playable.any(): 
            legal_moves = playable
        else: 
            legal_moves = player_hand
    
    return legal_moves


from .cards import DECK, INDEX_TO_SUIT, Rank

# Define the absolute hierarchy for non-trump cards (A > 10 > K > O > U > 9 > 8 > 7)
# Higher number = stronger card.
RANK_HIERARCHY = {
    "A": 7, "10": 6, "K": 5, "O": 4, "U": 3, "9": 2, "8": 1, "7": 0
}

# Precompute for O(1) lookup
CARD_STRENGTH = tuple(RANK_HIERARCHY[card[1:] if not card.startswith("10") else "10"] for card in DECK)

def determine_winner(trick: Trick, contract: GameContract) -> int: 
    """
    Determines the winner of a trick (complete or incomplete).
    
    1. If trumps were played: The lowest index in DECK (strongest trump) wins.
    2. If no trumps played: The highest rank of the lead suit wins.
    """
    if not trick.card_indices:
        raise ValueError("Cannot determine a winner for an empty trick.")

    # Get the indices of cards played and the corresponding player IDs
    played_indices = trick.card_indices
    player_ids = trick.player_ids[:len(played_indices)]

    # 1. Check for Trumps
    # Create a mask of which played cards are trumps
    is_trump_mask = [contract.trumpf_cards[idx].item() for idx in played_indices]
    
    if any(is_trump_mask):
        # Filter played cards to only those that are trumps
        # We want the one with the LOWEST index in DECK (Obers < Unters < Herz)
        best_idx = 999
        winner_id = -1
        
        for i, is_trump in enumerate(is_trump_mask):
            if is_trump:
                card_idx = played_indices[i]
                if card_idx < best_idx:
                    best_idx = card_idx
                    winner_id = player_ids[i]
        return winner_id

    # 2. No Trumps Played: Lead Suit Logic
    lead_card_idx = played_indices[0]
    lead_suit = INDEX_TO_SUIT[lead_card_idx]
    
    print(CARD_STRENGTH)
    
    best_strength = -1
    winner_id = -1

    for i, card_idx in enumerate(played_indices):
        # Only cards matching the lead suit can win
        if INDEX_TO_SUIT[card_idx] == lead_suit:
            strength = CARD_STRENGTH[card_idx]
            if strength > best_strength:
                best_strength = strength
                winner_id = player_ids[i]
                
    return winner_id


def get_playable_suits(player_hand: Cards,  trumpf_cards: Cards): 
    # one can only play if they have the color they want to play and not the corresponding ace
    non_trumpf_cards : Cards = player_hand & ~trumpf_cards
    
    playable_suits = []
    
    for suit in Suit:
        suit_cards = non_trumpf_cards & SUIT_MASKS[suit]
        if (not (suit_cards & ASSE).any()) and (suit_cards & ~ASSE).any(): 
            playable_suits.append(suit)
    
    return playable_suits