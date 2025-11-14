import sf_quant.optimizer as sfo

from research.models import Constraint


def zero_beta() -> Constraint:
    return Constraint(
        name=zero_beta.__name__, constraint=sfo.ZeroBeta(), columns=["predicted_beta"]
    )


def get_constraint(constraint_name: str) -> Constraint:
    match constraint_name:
        case "zero-beta":
            return zero_beta()
