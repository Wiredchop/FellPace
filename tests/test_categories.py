import pytest
import numpy as np
import pandas as pd
from fellpace.convert_tools import convert_categories

def test_categories():
    
    data = pd.read_csv(r"csv/ShefHalf-2021.csv")
    data['Cat.'].iloc[0] = 'M'
    data['Cat.'].iloc[1] = 'FSENIOR'
    convert_categories(data['Cat.'])