import pytest
from omados.engine.modes import GameModeFactory
from omados.engine.cards import Suit, Cards

@pytest.fixture
def sauspiel_eichel():
    """Returns a Sauspiel contract with Eichel as the called suit."""
    # Assuming caller_id=0 for these tests
    return GameModeFactory.create_sauspiel(suit=Suit.Eichel, caller_id=0)

@pytest.fixture
def rufsau_hand():
    """A hand containing the Rufsau (EA) and many Eichel cards (6 total)."""
    # Note: EO, EU are Trumps. EA, EK, E9, E8 are Eichel 'Farbe'.
    return Cards(['EO', 'EU', 'EA', 'S10', 'EK', 'E9', 'E8', 'S7'])

@pytest.fixture
def rufsau_hand_short():
    """A hand containing the Rufsau but only 3 Eichel cards total (cannot run away)."""
    return Cards(['EO', 'EA', 'EK', 'S10', 'S9', 'S8', 'S7', 'G7'])