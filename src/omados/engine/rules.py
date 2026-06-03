from .cards import ASSE, DECK, INDEX_TO_SUIT, SUIT_MASKS, Cards, Suit
from .modes import GameContract
from .tricks import Trick


def is_sauspiel(contract: GameContract) -> bool:
    return contract.spielart == "Sauspiel"


def has_rufsau(player_hand: Cards, contract: GameContract) -> bool:
    """Checks if the player has the Rufsau card in their hand."""
    return (player_hand & contract.rufsau).any()


def can_run_away(
    player_hand: Cards,
    lead_card_idx: int | None,
    contract: GameContract,
    ran_away: bool,
) -> bool:
    """Determines if the player can 'run away' from the Rufsau suit."""
    # Player must be the trick lead and not have already run away
    if (lead_card_idx is not None) or ran_away:
        return False

    # Game must be Sauspiel
    if contract.spielart != "Sauspiel":
        return False

    # Player must have at least four cards of the Rufsau suit to be able to run away
    assert contract.rufsau_suit is not None, "Rufsau suit must be defined for Sauspiel."
    rufsau_suit_cards = player_hand & SUIT_MASKS[contract.rufsau_suit]
    return rufsau_suit_cards.sum().item() >= 4


def is_running_away(
    player_hand: Cards,
    lead_card_idx: int | None,
    played_card_idx: int,
    contract: GameContract,
    ran_away: bool,
) -> bool:
    """Checks if the player is 'running away' from the Rufsau suit by playing a card
    that is not of the Rufsau suit."""

    if lead_card_idx is not None:
        return False  # Not the lead player, so can't be running away

    if not is_sauspiel(contract):
        return False  # Not a Sauspiel game, so this rule doesn't apply

    if ran_away:
        return False  # Player has already run away, so this check is irrelevant

    if not has_rufsau(player_hand, contract):
        return False  # Player doesn't have the Rufsau card, so they can't run away

    if not can_run_away(player_hand, lead_card_idx, contract, ran_away=ran_away):
        return False  # Player is not allowed to run away, so this check is irrelevant

    return (
        INDEX_TO_SUIT[played_card_idx] == contract.rufsau_suit
        and not contract.rufsau[played_card_idx].item()
    )


def get_legal_moves(
    player_hand: Cards,
    lead_card_idx: int | None,
    contract: GameContract,
    ran_away: bool = False,
) -> Cards:
    """Returns a mask of playable cards based on Schafkopf rules."""
    trumpf = contract.trumpf_cards

    # 1. Lead Player (First in trick)
    # check if the player can 'run away' from the Rufsau suit (Sauspiel specific rule)
    if lead_card_idx is None:
        if (
            is_sauspiel(contract)
            and has_rufsau(player_hand, contract)
            and (not can_run_away(player_hand, lead_card_idx, contract, ran_away))
            and (not ran_away)
        ):
            assert contract.rufsau_suit is not None, (
                "Rufsau suit must be defined if Rufsau card is in hand."
            )
            return (player_hand & ~SUIT_MASKS[contract.rufsau_suit]) | contract.rufsau
        else:
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

    # if player did not ran away and the lead suit (non-trumpf) is of the rufsau suit,
    # they can only play that.
    if not ran_away:
        if playable.any():
            assert contract.rufsau_suit is not None, (
                "Rufsau suit must be defined if there are playable cards."
            )
            legal_moves = playable & (
                contract.rufsau | ~SUIT_MASKS[contract.rufsau_suit]
            )
        else:
            legal_moves = player_hand & ~contract.rufsau
    else:
        legal_moves = playable if playable.any() else player_hand

    return legal_moves


# Define the absolute hierarchy for non-trump cards (A > 10 > K > O > U > 9 > 8 > 7)
# Higher number = stronger card.
RANK_HIERARCHY = {"A": 7, "10": 6, "K": 5, "O": 4, "U": 3, "9": 2, "8": 1, "7": 0}

# Precompute for O(1) lookup
CARD_STRENGTH = tuple(
    RANK_HIERARCHY[card[1:] if not card.startswith("10") else "10"] for card in DECK
)


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
    player_ids = trick.player_ids[: len(played_indices)]

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


def get_playable_suits(player_hand: Cards, trumpf_cards: Cards) -> list[Suit]:
    # one can only play if they have the color they want to play and not the
    # corresponding ace
    non_trumpf_cards: Cards = player_hand & ~trumpf_cards

    playable_suits = []

    for suit in Suit:
        suit_cards = non_trumpf_cards & SUIT_MASKS[suit]
        if (not (suit_cards & ASSE).any()) and (suit_cards & ~ASSE).any():
            playable_suits.append(suit)

    return playable_suits
