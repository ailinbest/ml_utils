import ml_utils
import pandas as pd
import numpy as np
from sklearn import datasets


def test_describe_missing():
    df = pd.DataFrame(datasets.load_boston().get('data'))
    print(ml_utils.describe_missing(df=df, include='float'))


if __name__ == '__main__':
    test_describe_missing()
