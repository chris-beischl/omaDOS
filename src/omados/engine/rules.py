import torch
from .cards import Cards, SUIT_MASKS, ASSE, SUIT_LOOKUP, CARD_POINTS, INDEX_TO_SUIT, Suit
from .modes import GameContract

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

# TODO: totally rework
def determine_winner(trick_cards_idx: list[int], leader_id: int, contract: GameContract) -> int:
    """Returns the absolute player_id (0-3) who won the trick."""
    trumpf = contract.trumpf_cards
    
    best_card_idx = trick_cards_idx[0]
    winner_local_idx = 0
    
    lead_card = trick_cards_idx[0]
    is_lead_trumpf = trumpf[lead_card]
    lead_suit = INDEX_TO_SUIT[lead_card]

    for i in range(1, 4):
        current_card = trick_cards_idx[i]
        
        # 1. If current is trump and best wasn't -> current wins
        if trumpf[current_card] and not trumpf[best_card_idx]:
            best_card_idx = current_card
            winner_local_idx = i
        # 2. If both are trump -> higher index in deck wins (since O/U/H7 are early in your DECK)
        elif trumpf[current_card] and trumpf[best_card_idx]:
            if current_card < best_card_idx: # Your DECK order: Obers(0-3), Unters(4-7)...
                best_card_idx = current_card
                winner_local_idx = i
        # 3. If neither is trump -> must match lead suit and be higher rank (lower index)
        elif not trumpf[current_card] and not trumpf[best_card_idx]:
            if INDEX_TO_SUIT[current_card] == lead_suit and current_card < best_card_idx:
                best_card_idx = current_card
                winner_local_idx = i

    return (leader_id + winner_local_idx) % 4

# TODO: check with all rules!
def get_playable_suits(player_hand: Cards,  trumpf_cards: Cards): 
    # one can only play if they have the color they want to play and not the corresponding ace
    non_trumpf_cards : Cards = player_hand & ~trumpf_cards
    
    playable_suits = []
    
    for suit in Suit:
        suit_cards = non_trumpf_cards & SUIT_MASKS[suit]
        if (not (suit_cards & ASSE).any()) and (suit_cards & ~ASSE).any(): 
            playable_suits.append(suit)
    
    return playable_suits