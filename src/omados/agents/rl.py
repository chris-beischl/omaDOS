import random

import torch

from omados.engine.cards import Cards
from omados.engine.modes import SPIELART_RANK, GameContract, Spielart
from omados.engine.rules import get_available_games
from omados.engine.tricks import Trick
from omados.models.policy_net import PolicyNet

from .base import BaseAgent, BaseInformationState, BiddingState

OBS_SIZE = 1296


class RLAgent(BaseAgent):
    def __init__(self, player_id: int, model: PolicyNet):
        super().__init__(player_id)
        self.model = model
        self.trajectory: list[dict[str, torch.Tensor]] = []

    def obs_tensor(self, state: BaseInformationState) -> torch.Tensor:
        # derive played_cards from self.memory (past tricks)
        # concatenate all components into one flat tensor
        hand_tensor = state.hand.cards
        legal_moves_tensor = state.legal_moves.cards
        trumpf_cards_tensor = state.contract.trumpf_cards.cards
        rufsau_tensor = state.contract.rufsau.cards
        spielart_tensor = torch.nn.functional.one_hot(
            torch.tensor(SPIELART_RANK[state.contract.spielart], dtype=torch.long),
            num_classes=8,
        )
        player_id_tensor = torch.nn.functional.one_hot(
            torch.tensor(self.player_id), num_classes=4
        )
        caller_id_tensor = torch.nn.functional.one_hot(
            torch.tensor(state.contract.caller_id, dtype=torch.long), num_classes=4
        )

        current_trick_cards_tensor = state.current_trick.cards_tensor.flatten()  # → 128
        current_trick_players_tensor = (
            state.current_trick.players_tensor.flatten()
        )  # → 16

        past_cards_tensor = torch.zeros(7, 4, 32)
        past_players_tensor = torch.zeros(7, 4, 4)
        for i, entry in enumerate(self.memory):
            trick = entry["trick"]
            past_cards_tensor[i] = trick.cards_tensor
            past_players_tensor[i] = trick.players_tensor

        past_cards_tensor = past_cards_tensor.flatten()
        past_players_tensor = past_players_tensor.flatten()

        obs = torch.cat(
            [
                hand_tensor.float(),
                legal_moves_tensor.float(),
                trumpf_cards_tensor.float(),
                rufsau_tensor.float(),
                spielart_tensor,
                player_id_tensor,
                caller_id_tensor,
                current_trick_cards_tensor,
                current_trick_players_tensor,
                past_cards_tensor,
                past_players_tensor,
            ]
        )
        assert obs.shape[0] == OBS_SIZE, (
            f"obs size mismatch: {obs.shape[0]} != {OBS_SIZE}"
        )
        return obs

    # TODO: remove once proper bidding logic is implemented!
    def decide_bidding(self, hand: Cards) -> GameContract | None:
        pass

    def wants_to_play(
        self, hand: Cards, state: BiddingState, viable_spielarten: list[Spielart]
    ) -> bool:
        # If someone already declared → pass
        if any(state.declarations.values()):
            return False
        return len(viable_spielarten) > 0 and random.random() < 0.5

    def make_bid(
        self, hand: Cards, state: BiddingState, viable_spielarten: list[Spielart]
    ) -> Spielart:
        return viable_spielarten[0] if viable_spielarten else Spielart.Wenz

    def make_bid_or_fold(
        self, hand: Cards, state: BiddingState, viable_spielarten: list[Spielart]
    ) -> Spielart | None:
        return None

    def wants_to_continue(
        self, hand: Cards, state: BiddingState, viable_spielarten: list[Spielart]
    ) -> bool:
        return False

    def declare_final_game(
        self, hand: Cards, state: BiddingState, viable_spielarten: list[Spielart]
    ) -> GameContract:
        available = get_available_games(hand, self.player_id)
        # Always prefer Sauspiel
        sauspiele = [
            g
            for g in available
            if g.spielart == Spielart.Sauspiel and g.spielart in viable_spielarten
        ]
        if sauspiele:
            return random.choice(sauspiele)
        # Fallback: lowest viable game
        viable = [g for g in available if g.spielart in viable_spielarten]
        return viable[0] if viable else available[0]

    def wants_to_give_contra(self, contract: GameContract, first_trick: Trick) -> bool:
        return False

    def wants_to_give_retour(self, contract: GameContract, first_trick: Trick) -> bool:
        return False

    def play_card(self, state: BaseInformationState) -> int:
        obs = self.obs_tensor(state)

        if self.model.training:
            logits = self.model(obs)
        else:
            with torch.no_grad():
                logits = self.model(obs)

        mask = state.legal_moves.cards  # bool tensor, shape (32,)
        logits = logits.clone()
        logits[~mask] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        action = int(torch.multinomial(probs, 1).item())

        if self.model.training:
            log_prob = torch.log(probs[action])
            self.trajectory.append({"obs": obs, "log_prob": log_prob})

        return action

    def observe_trick_results(self, trick: Trick, winner_id: int) -> None:
        """Called by the Env after every trick so the agent can learn/remember."""
        self.memory.append({"trick": trick, "winner": winner_id})
