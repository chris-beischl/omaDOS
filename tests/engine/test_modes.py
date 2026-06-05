"""
Tests for get_trumpf_cards, GameContract.__post_init__, SPIELART_RANK,
and GameModeFactory in modes.py.
"""

import pytest

from omados.engine.cards import HERZ, OBER, UNTER, Cards, Suit
from omados.engine.modes import (
    SPIELART_RANK,
    GameModeFactory,
    Spielart,
    get_trumpf_cards,
)

# ---------------------------------------------------------------------------
# get_trumpf_cards — card counts and membership
# ---------------------------------------------------------------------------


def test_sauspiel_trumpf_is_obers_unters_herz():
    trumpf = get_trumpf_cards(Spielart.Sauspiel)
    assert trumpf.sum() == 14  # 4 Obers + 4 Unters + 6 Herz non-O/U
    assert (trumpf & OBER).to_list() == OBER.to_list()  # all Obers are trump
    assert (trumpf & UNTER).to_list() == UNTER.to_list()  # all Unters are trump
    assert (trumpf & HERZ).to_list() == HERZ.to_list()  # all Herz are trump


def test_ramsch_same_as_sauspiel():
    """Ramsch uses the same trump set as Sauspiel."""
    assert (
        get_trumpf_cards(Spielart.Ramsch).to_list()
        == get_trumpf_cards(Spielart.Sauspiel).to_list()
    )


def test_wenz_trumpf_is_only_unters():
    trumpf = get_trumpf_cards(Spielart.Wenz)
    assert trumpf.sum() == 4
    assert (trumpf & UNTER).to_list() == UNTER.to_list()
    assert not (trumpf & OBER).any()
    assert not (trumpf & HERZ & ~UNTER).any()


def test_geier_trumpf_is_only_obers():
    trumpf = get_trumpf_cards(Spielart.Geier)
    assert trumpf.sum() == 4
    assert (trumpf & OBER).to_list() == OBER.to_list()
    assert not (trumpf & UNTER).any()


def test_farbwenz_includes_unters_and_suit():
    trumpf = get_trumpf_cards(Spielart.Farbwenz, suit=Suit.Eichel)
    # UNTER (4) | SUIT_MASKS[Eichel] (8) — EU counted once = 11 total
    assert trumpf.sum() == 11
    assert (trumpf & UNTER).to_list() == UNTER.to_list()
    # EO (Ober) IS included because it's part of SUIT_MASKS[Eichel]
    eo = Cards(["EO"])
    assert (trumpf & eo).any()


def test_farbwenz_requires_suit():
    with pytest.raises((AssertionError, TypeError)):
        get_trumpf_cards(Spielart.Farbwenz)  # suit=None should fail


def test_farbgeier_includes_obers_and_suit():
    trumpf = get_trumpf_cards(Spielart.Farbgeier, suit=Suit.Gras)
    # OBER (4) | SUIT_MASKS[Gras] (8) — GO counted once = 11 total
    assert trumpf.sum() == 11
    assert (trumpf & OBER).to_list() == OBER.to_list()
    gu = Cards(["GU"])
    assert (trumpf & gu).any()  # GU is in SUIT_MASKS[Gras] → included


def test_solo_herz_includes_obers_unters_herz():
    trumpf = get_trumpf_cards(Spielart.Solo, suit=Suit.Herz)
    # Same as Sauspiel: OO+UU+Herz = 14
    assert trumpf.sum() == 14
    assert trumpf.to_list() == get_trumpf_cards(Spielart.Sauspiel).to_list()


def test_solo_eichel_includes_obers_unters_eichel():
    trumpf = get_trumpf_cards(Spielart.Solo, suit=Suit.Eichel)
    # 4 Obers + 4 Unters + 6 Eichel non-O/U = 14
    assert trumpf.sum() == 14
    assert (trumpf & OBER).to_list() == OBER.to_list()
    assert (trumpf & UNTER).to_list() == UNTER.to_list()
    # Herz non-trump cards should not appear
    herz_farbe = Cards(["HA", "H10", "HK", "H9", "H8", "H7"])
    assert not (trumpf & herz_farbe).any()


def test_solo_requires_suit():
    with pytest.raises((AssertionError, TypeError)):
        get_trumpf_cards(Spielart.Solo)


def test_sie_same_as_sauspiel():
    """Sie uses standard Sauspiel trumps."""
    assert (
        get_trumpf_cards(Spielart.Sie).to_list()
        == get_trumpf_cards(Spielart.Sauspiel).to_list()
    )


# ---------------------------------------------------------------------------
# GameContract.__post_init__ — rufsau computation
# ---------------------------------------------------------------------------


def test_gamecontract_sauspiel_rufsau_is_ace():
    contract = GameModeFactory.create_sauspiel(caller_id=0, suit=Suit.Eichel)
    assert contract.rufsau.sum() == 1
    assert "EA" in contract.rufsau.to_list()


def test_gamecontract_sauspiel_gras_rufsau():
    contract = GameModeFactory.create_sauspiel(caller_id=0, suit=Suit.Gras)
    assert "GA" in contract.rufsau.to_list()
    assert "EA" not in contract.rufsau.to_list()


def test_gamecontract_non_sauspiel_rufsau_is_empty():
    """Wenz, Solo, Geier should have no rufsau."""
    for contract in [
        GameModeFactory.create_wenz(caller_id=0),
        GameModeFactory.create_solo(caller_id=0, suit=Suit.Herz),
        GameModeFactory.create_geier(caller_id=0),
    ]:
        assert not contract.rufsau.any(), (
            f"{contract.spielart} should have empty rufsau"
        )


def test_gamecontract_rufsau_suit_stored_correctly():
    contract = GameModeFactory.create_sauspiel(caller_id=2, suit=Suit.Schellen)
    assert contract.rufsau_suit == Suit.Schellen
    assert contract.caller_id == 2


# ---------------------------------------------------------------------------
# SPIELART_RANK — ordering invariants
# ---------------------------------------------------------------------------


def test_spielart_rank_sauspiel_is_lowest():
    assert SPIELART_RANK[Spielart.Sauspiel] == 0


def test_spielart_rank_sie_is_highest():
    assert SPIELART_RANK[Spielart.Sie] == max(SPIELART_RANK.values())


def test_spielart_rank_ordering():
    """Solo must outrank Wenz, Wenz must outrank Sauspiel, etc."""
    assert SPIELART_RANK[Spielart.Solo] > SPIELART_RANK[Spielart.Wenz]
    assert SPIELART_RANK[Spielart.Wenz] > SPIELART_RANK[Spielart.Geier]
    assert SPIELART_RANK[Spielart.Geier] > SPIELART_RANK[Spielart.Farbwenz]
    assert SPIELART_RANK[Spielart.Farbwenz] > SPIELART_RANK[Spielart.Farbgeier]
    assert SPIELART_RANK[Spielart.Farbgeier] > SPIELART_RANK[Spielart.Sauspiel]


def test_spielart_rank_all_unique():
    values = list(SPIELART_RANK.values())
    assert len(values) == len(set(values))


def test_spielart_rank_covers_all_spielarten():
    """Every Spielart that can be bid must have a rank entry."""
    biddable = {
        Spielart.Sauspiel,
        Spielart.Farbgeier,
        Spielart.Farbwenz,
        Spielart.Geier,
        Spielart.Wenz,
        Spielart.Solo,
        Spielart.Sie,
    }
    assert biddable <= set(SPIELART_RANK.keys())


# ---------------------------------------------------------------------------
# GameModeFactory — smoke tests
# ---------------------------------------------------------------------------


def test_factory_sauspiel():
    c = GameModeFactory.create_sauspiel(caller_id=1, suit=Suit.Gras)
    assert c.spielart == Spielart.Sauspiel
    assert c.caller_id == 1
    assert c.rufsau_suit == Suit.Gras


def test_factory_wenz_pure():
    c = GameModeFactory.create_wenz(caller_id=0)
    assert c.spielart == Spielart.Wenz
    assert c.rufsau_suit is None


def test_factory_wenz_farb():
    c = GameModeFactory.create_wenz(caller_id=0, suit=Suit.Schellen)
    assert c.spielart == Spielart.Farbwenz


def test_factory_geier_pure():
    c = GameModeFactory.create_geier(caller_id=0)
    assert c.spielart == Spielart.Geier


def test_factory_geier_farb():
    c = GameModeFactory.create_geier(caller_id=0, suit=Suit.Eichel)
    assert c.spielart == Spielart.Farbgeier


def test_factory_solo():
    c = GameModeFactory.create_solo(caller_id=3, suit=Suit.Schellen)
    assert c.spielart == Spielart.Solo
    assert c.caller_id == 3


def test_factory_ramsch():
    c = GameModeFactory.create_ramsch()
    assert c.spielart == Spielart.Ramsch
    assert c.caller_id == -1
