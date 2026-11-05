from .cbba import CBBA, CBBA_ETC
from .greedy_auction import GreedyAuction
from .hungarian import HungarianAssignment
from .nearest import NearestFeasible

CENTRALIZED = {
    "nearest": NearestFeasible,
    "greedy": GreedyAuction,
    "hungarian": HungarianAssignment,
    "cbba": CBBA,
    "cbba_etc": CBBA_ETC,
}
