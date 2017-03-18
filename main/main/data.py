## This file will take care of dataset building

import pandas as pd
import numpy as np


def dataset_building():
    datafile = pd.ExcelFile('exchange.xlsx')
    data = datafile.parse('exchange')
    return data


