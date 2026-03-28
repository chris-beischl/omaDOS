import torch


class Stich:
    def __init__(self):
        # karten: 8 Stiche, 4 Spieler, 32 Karten (one hot). Noch nicht gespielte Stiche sind mit 0 gefüllt. Starten immer beim Rauskommer des Stiches
        # erst vollständige stiche werden hinzugefügt
        self.karten: torch.Tensor = torch.zeros((4, 32))
        # welcher spieler die karten des stiches gespielt hat, absolut zum ersten Rauskommer
        self.spieler: torch.Tensor = torch.zeros((4, 4))
        # welcher spieler den stich gewonnen hat, absolut zum ersten Rauskommer
        self.gewinner: torch.Tensor = torch.zeros(4)

    def flatten(self):
        return torch.cat([self.karten.flatten(), self.spieler.flatten(), self.gewinner.flatten()])

    @staticmethod
    def flat_shape():
        return 32 * 4 + 4 * 4 + 4


class PublicGameState:
    def __init__(self, spieler_id, farb_id):
        # spieler: one hot: spieler_id relativ zum Rauskommer (0-3, 0=Rauskommer)
        self.spieler: torch.Tensor = torch.nn.functional.one_hot(torch.tensor(spieler_id), num_classes=4).float()
        # gesuchte_farbe: one hot: farb_id (0-2, 0=Schellen, 1=Gras, 2=Eichel)
        self.gesuchte_farbe: torch.Tensor = torch.nn.functional.one_hot(torch.tensor(farb_id), num_classes=3).float()  # one hot
        self.__stiche: list[Stich] = []
        # aktueller stich: one hot: die bisher gelegten karten im aktuellen stich, 4 Spieler, 32 Karten (one hot). Starten immer beim Rauskommer des Stiches
        self.aktueller_stich: Stich = Stich()
        self.aktuelle_augen: torch.Tensor = torch.zeros(4)  # points per player [0,120] maps to [0,1]

    def stiche_flat(self):
        output = torch.zeros((8, Stich.flat_shape()))
        for i, stich in enumerate(self.__stiche):
            output[i] = stich.flat()
        return output.flatten()

    def flatten(self):
        return torch.cat([self.spieler.flatten(), self.gesuchte_farbe.flatten(), self.stiche_flat(), self.aktueller_stich.flat(), self.aktuelle_augen.flatten()])


class Player:
    def __init__(self, spieler_id):
        self.hände: torch.Tensor = torch.zeros((4, 32))  # 8 hot cards per player, the cards of the other players are all cards they could have
        self.position: torch.Tensor = torch.nn.functional.one_hot(torch.tensor(spieler_id), num_classes=4).float()  # one hot: absolut zum ersten Rauskommer

    def flatten(self):
        return torch.cat([self.hände.flatten(), self.position.flatten()])


class Game:
    def __init__(self, spieler_id, farb_id):
        # TODO: chose reasonable init params
        self.public_state = PublicGameState(spieler_id, farb_id)
        self.players = [Player(i) for i in range(4)]
