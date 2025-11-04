from models import Dataset

def crsp() -> Dataset:
    return Dataset(
        name='crsp',
        primary_keys=['date', 'permno']
    )

def get_dataset(name: str) -> Dataset:
    match name:
        case 'crsp':
            return crsp()
        case _:
            raise ValueError