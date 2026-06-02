import pytest

from omados.engine.cards import Suit
from omados.engine.modes import GameModeFactory
from omados.engine.rules import determine_winner
from omados.engine.tricks import Trick


@pytest.fixture
def sauspiel_eichel():
    return GameModeFactory.create_sauspiel(caller_id=0, suit=Suit.Eichel)


@pytest.fixture
def wenz():
    return GameModeFactory.create_wenz(caller_id=0)


# --- Incomplete Trick Scenarios ---


def test_winner_incomplete_trick(sauspiel_eichel):
    """Verify the winner updates correctly as an incomplete trick progresses."""
    trick = Trick(lead_player_id=1)

    # 1st card: P1 leads Gras 7
    trick.add_card("G7")
    assert determine_winner(trick, sauspiel_eichel) == 1

    # 2nd card: P2 plays Gras Ace (takes the lead)
    trick.add_card("GA")
    assert determine_winner(trick, sauspiel_eichel) == 2

    # 3rd card: P3 plays Schellen 7 (wrong suit, P2 still winning)
    trick.add_card("S7")
    assert determine_winner(trick, sauspiel_eichel) == 2

    # 4th card: P0 plays Herz 7 (Trump! P0 takes the trick)
    trick.add_card("H7")
    assert determine_winner(trick, sauspiel_eichel) == 0


# --- Full Trick Scenarios (Sauspiel) ---


def test_winner_simple_suit_follow(sauspiel_eichel):
    """Lead G7, others follow with higher Gras cards. No trumps."""
    # Player 2 leads
    cards = ["G7", "G10", "GK", "G9"]
    trick = Trick(lead_player_id=2, cards=cards)

    # Order: P2(G7), P3(G10), P0(GK), P1(G9). P3 played G10 (highest).
    assert determine_winner(trick, sauspiel_eichel) == 3


def test_winner_trump_beats_suit(sauspiel_eichel):
    """Lead Eichel Ace, someone trumps with a Herz card."""
    cards = ["EA", "H7", "EK", "E9"]
    trick = Trick(lead_player_id=0, cards=cards)

    # Player 1 (H7) should win even though H7 is worth 0 points compared to EA's 11
    assert determine_winner(trick, sauspiel_eichel) == 1


def test_winner_overtrump_with_ober(sauspiel_eichel):
    """Multiple trumps played, highest one wins."""
    # P1(HA), P2(SU), P3(GO), P0(H10)
    cards = ["HA", "SU", "GO", "H10"]
    trick = Trick(lead_player_id=1, cards=cards)

    # Player 3 played the Gras Ober (highest trump in this trick)
    assert determine_winner(trick, sauspiel_eichel) == 3


def test_winner_discard_wrong_suit(sauspiel_eichel):
    """Player cannot follow suit and discards a non-trump."""
    # P0(E10), P1(EK), P2(SA - wrong suit), P3(E7)
    cards = ["E10", "EK", "SA", "E7"]
    trick = Trick(lead_player_id=0, cards=cards)

    # Player 0's E10 wins; P2's SA is high but wrong suit
    assert determine_winner(trick, sauspiel_eichel) == 0


# --- Wenz Specific Scenarios ---


def test_winner_wenz_ober_is_not_trump(wenz):
    """In Wenz, Obers are just normal suit cards."""
    # P0 leads Eichel Ober. P1(EA), P2(HU - Trump), P3(E7)
    cards = ["EO", "EA", "HU", "E7"]
    trick = Trick(lead_player_id=0, cards=cards)

    # In Sauspiel, EO would win. In Wenz, HU is the only trump.
    assert determine_winner(trick, wenz) == 2


def test_winner_wenz_suit_logic(wenz):
    """In Wenz, leading an Ober means others must follow that suit."""
    # Lead Eichel Ober. P1(E7), P2(E10), P3(E8)
    cards = ["EO", "E7", "E10", "E8"]
    trick = Trick(lead_player_id=0, cards=cards)

    # EO is the highest Eichel card now that Unters are removed from the suit
    assert determine_winner(trick, wenz) == 2


def test_winner_wenz_suit_order(wenz):
    """In Wenz, an Ober is just a normal suit card, weaker than A, 10, K."""
    # Player 0 leads Eichel Ober (Strength 3)
    # Player 1 plays Eichel Ace (Strength 11)
    # Player 2 plays Eichel 10 (Strength 10)
    # Player 3 plays Eichel 9 (Strength 0)
    cards = ["EO", "EA", "E10", "E9"]
    trick = Trick(lead_player_id=0, cards=cards)

    # Player 1 played the Ace, which correctly beats the Ober in a Wenz!
    assert determine_winner(trick, wenz) == 1


def test_winner_geier_suit_order():
    """In Geier, an Unter is a normal suit card, weaker than A, 10, K, O."""
    geier = GameModeFactory.create_geier(caller_id=0)

    # Player 2 leads Gras Unter (Strength 3)
    # Player 3 plays Gras Ober (Strength 4)
    # Player 0 plays Gras King (Strength 5)
    # Player 1 plays Gras 7 (Strength 0)
    cards = ["GU", "GO", "GK", "G7"]
    trick = Trick(lead_player_id=2, cards=cards)

    assert determine_winner(trick, geier) == 3


# --- Edge Cases ---


def test_empty_trick_raises_error(sauspiel_eichel):
    trick = Trick(lead_player_id=0)
    with pytest.raises(ValueError, match="empty trick"):
        determine_winner(trick, sauspiel_eichel)
