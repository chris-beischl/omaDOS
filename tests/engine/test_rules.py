from omados.engine.cards import CARD_TO_INDEX, Cards
from omados.engine.rules import get_legal_moves


def test_sauspiel_lead_can_run_away(sauspiel_eichel, rufsau_hand):
    """
    If lead player has 4+ cards of the called suit (including Ace),
    they are allowed to lead a small card of that suit (Davonlaufen).
    """
    legal = get_legal_moves(rufsau_hand, lead_card_idx=None, contract=sauspiel_eichel)

    # Should be able to play everything (including E9, EK, etc.)
    assert legal.sum() == rufsau_hand.sum()
    assert "E9" in legal.to_list()


def test_sauspiel_lead_cannot_run_away_if_short(sauspiel_eichel, rufsau_hand_short):
    """
    If lead player has < 4 cards of called suit, they CANNOT lead a small card
    of that suit. They must lead the Ace or something else.
    """
    legal = get_legal_moves(
        rufsau_hand_short, lead_card_idx=None, contract=sauspiel_eichel
    )

    # Must NOT contain Eichel King (EK) if it's the called suit and we are short
    assert "EK" not in legal.to_list()
    # But the Ace (EA) itself is allowed to be led
    assert "EA" in legal.to_list()


def test_sauspiel_follower_must_play_ace(sauspiel_eichel, rufsau_hand):
    """If the called suit is led, the player MUST play the Rufsau Ace."""
    e7_idx = CARD_TO_INDEX["E7"]
    legal = get_legal_moves(rufsau_hand, lead_card_idx=e7_idx, contract=sauspiel_eichel)

    # Out of the Eichel cards, ONLY the Ace (EA) should be playable
    assert legal.sum() == 1
    assert legal.to_list() == ["EA"]


def test_sauspiel_follower_davongelaufen(sauspiel_eichel, rufsau_hand):
    """If the player already 'ran away', they no longer have to play the Ace."""
    e7_idx = CARD_TO_INDEX["E7"]
    # Removing E8 to simulate one card already played from your snippet logic
    hand = Cards(["EO", "EU", "EA", "S10", "EK", "E9", "S7"])

    legal = get_legal_moves(
        hand, lead_card_idx=e7_idx, contract=sauspiel_eichel, ran_away=True
    )

    # Now all Eichel 'Farbe' cards are playable (EA, EK, E9)
    assert "EK" in legal.to_list()
    assert "E9" in legal.to_list()
    assert "EA" in legal.to_list()


def test_sauspiel_free_cannot_discard_rufsau(sauspiel_eichel, rufsau_hand):
    """If a different suit (Gras) is led and the player is free, they cannot discard
    the Ace."""
    g7_idx = CARD_TO_INDEX["G7"]
    legal = get_legal_moves(rufsau_hand, lead_card_idx=g7_idx, contract=sauspiel_eichel)

    # Should be able to play everything EXCEPT the Rufsau Ace (EA)
    assert "EA" not in legal.to_list()
    assert "S10" in legal.to_list()
    assert legal.sum() == rufsau_hand.sum() - 1


def test_sauspiel_free_can_discard_rufsau_if_ran_away(sauspiel_eichel, rufsau_hand):
    """If player is free and already ran away, they ARE allowed to discard the Ace."""
    g7_idx = CARD_TO_INDEX["G7"]
    legal = get_legal_moves(
        rufsau_hand, lead_card_idx=g7_idx, contract=sauspiel_eichel, ran_away=True
    )

    assert "EA" in legal.to_list()
