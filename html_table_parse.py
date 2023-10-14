import requests
import pandas as pd
import numpy as np
from typing import Literal, Tuple, List
import numpy.typing as npt
from FellPace_tools import * 





url = "https://www.fellrunner.org.uk/results/2ec842a9-012a-43df-afe3-5dd365cf2ca5"
scraped_data,_ = get_table_from_URL(url)

(race_metadata, entries) = process_data_for_DB(scraped_data)
            
        

#For testing, write the table to text so we can use it in other area

entries.data.to_csv(race_meta.race_name + '.csv',index=False)





