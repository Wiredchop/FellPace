import pandas as pd
import requests


def get_racetek_api(URL: str) -> pd.DataFrame:
    response = requests.get(URL)
    # racetek API gives a list of entries (each a single list).
    fulljson = response.json()
    data = pd.DataFrame(data = fulljson,
                        columns = ['No.','First Name','Surname','Gender','Age Cat','Course','Club','Time',''],
                        )
    data['Position']=data.index
    return data


def get_avtiming_api(URL: str) -> pd.DataFrame:
    api_url = URL
    response = requests.get(api_url)
    fulljson = response.json()
    data = fulljson['data']

    #These aren't consistent but should be ok to spoof column headings, will automatically pad column names according to the size of the array
    col_names = ['Bib','Pos','Name','M/F','Cat','Cat_Pos','Club','Chip','Time']
    pads_needed = len(data[0]) - len(col_names)
    if pads_needed > 0:
        for i in range(pads_needed):
            col_names.append(f'Pad{i}')

    return pd.DataFrame(data,columns=col_names)