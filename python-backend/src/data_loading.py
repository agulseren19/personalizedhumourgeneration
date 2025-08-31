from datasets import Dataset, DatasetDict, Features, Value, ClassLabel
import pandas as pd
import pathlib as p

_root = p.Path(__file__).resolve().parents[1] / 'data/processed'

def _load_parquet(name):
    return Dataset.from_pandas(pd.read_parquet(_root / name),
                             preserve_index=False)

def load_dataset():
    ds = DatasetDict({
        'train': _load_parquet('cah_train.parquet'),
        'validation': _load_parquet('cah_valid.parquet'),
        'test': _load_parquet('cah_test.parquet')
    })
    # Cast dtypes for speed & memory
    ds = ds.cast_column('winner', ClassLabel(num_classes=2, names=['lose','win']))
    return ds 