import torch
from .cards import Cards, CARD_POINTS, CARD_TO_INDEX, DECK
from .modes import GameContract

# TODO: rework internal logic to more heavily utilize tensors? 
class Trick:
    def __init__(self, lead_player_id: int, cards: list[str | int | Cards] | None = None):
        """
        Initializes a Trick. Can be empty, partially played, or full.
        
        Args:
            lead_player_id: The global ID (0-3) of the player who starts the trick.
            cards: A list of card strings (e.g., 'EO'), integer indices, or single-card Cards objects.
        """
        self.lead_player_id = lead_player_id
        self.card_indices: list[int] = []
        
        # The sequence of players is fixed once the lead player is known
        self.player_ids = [(self.lead_player_id + i) % 4 for i in range(4)]
        
        # Pre-allocate ML tensors
        self.cards_tensor = torch.zeros((4, 32), dtype=torch.float32)
        self.players_tensor = self._init_players_tensor()
        
        # Populate if initial cards were provided
        if cards:
            for card in cards:
                self.add_card(card)

    def _init_players_tensor(self) -> torch.Tensor:
        """Immediately initializes the (4, 4) one-hot tensor for player order."""
        tensor = torch.zeros((4, 4), dtype=torch.float32)
        for turn_idx, player_id in enumerate(self.player_ids):
            tensor[turn_idx, player_id] = 1.0
        return tensor

    def add_card(self, card: str | int | Cards):
        """Updates the trick with a newly played card."""
        if self.is_full:
            raise ValueError("Trick is already full.")

        # 1. Resolve to an integer index
        if isinstance(card, str):
            idx = CARD_TO_INDEX[card]
            
        elif isinstance(card, Cards):
            # Verify this 'Cards' object represents exactly one card
            count = card.sum()
            if count != 1:
                raise ValueError(f"Can only play 1 card into a trick, but got {count}.")
            idx = int(torch.where(card.cards)[0].item())
            
        else:
            idx = int(card)

        # 2. Update state and tensors
        current_turn = len(self.card_indices)
        self.card_indices.append(idx)
        self.cards_tensor[current_turn, idx] = 1.0

    @property
    def is_full(self) -> bool:
        """Helper to quickly check if the trick has concluded."""
        return len(self.card_indices) == 4

    @property
    def points(self) -> int:
        """Returns the total points currently in the trick."""
        if not self.card_indices:
            return 0
        played_mask = self.cards_tensor.sum(dim=0)
        return int(torch.dot(played_mask, CARD_POINTS.float()).item())

    def contains_trumpf(self, contract: GameContract) -> bool:
        """Checks if any currently played card is a trumpf."""
        if not self.card_indices:
            return False
        played_mask = self.cards_tensor.sum(dim=0).bool()
        return bool((played_mask & contract.trumpf_cards.cards).any().item())

    def __repr__(self) -> str:
        cards_str = [DECK[i] for i in self.card_indices]
        # Only show the players who have actually played a card so far
        played_by = self.player_ids[:len(self.card_indices)]
        return f"Trick(lead={self.lead_player_id}, cards={cards_str}, played_by={played_by})"