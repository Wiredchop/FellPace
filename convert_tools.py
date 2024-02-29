import numpy.typing as npt 
import numpy as np
import pandas as pd
import FellPace_tools
def time_string_to_seconds(duration_strings : npt.NDArray) -> npt.NDArray:   
    ftr = [3600,60,1]
    out =[]
    
    for duration_string in duration_strings:
        try:
            hrsminssecs = duration_string.split(':')
            if len(hrsminssecs) < 3: # Checking whether just 'mm:ss' modifying crudely
                hrsminssecs.insert(0,'00')
            # Removing any value after a decimal point as this can't be cast to an integer
            hrsminssecs[2] = hrsminssecs[2].rsplit('.')[0]
            out.append(sum([a*b for a,b in zip(ftr, [int(i) for i in hrsminssecs])]))
        except:
            out.append(None)
        
    return np.array(out)


from re import sub,match,search
class ParkRunConverter:
    
    def __init__(self, data: pd.DataFrame) -> None:
        names = self.convert_PR_name(data['parkrunner'])
        categories = self.convert_PR_categories(data['Age Group'])
        times = self.convert_PR_time(data['Time'])
        #Construct the data table
        self.entries = FellPace_tools.race_entries(len(data.index))
        self.entries.add_column_of_data('Position',data['Position'])
        self.entries.add_column_of_data('Club',data['Club'])
        self.entries.add_column_of_data('Racer_Name',names)
        #Adding directly as conversion already been completed so don't need inbuilt converter
        self.entries.data['Cat_Name'] = categories
        self.entries.data['Time'] = times
    
    def convert_PR_time(self,duration_strings : npt.ArrayLike) -> npt.ArrayLike:
        ftr = [3600,60,1]
        out = []
        for i,duration_string in enumerate(duration_strings):
            if type(duration_string) is not str:
                out.append(None) # Sometimes nan gets returned from the table
                continue
            duration_match = match('[0-9:]+',duration_string) # should only match first string because match
            if not duration_match:
                out.append(None)
                continue
            duration_string = duration_match.group()
            try:
                hrsminssecs = duration_string.split(':')
                if len(hrsminssecs) < 3: # Checking whether just 'mm:ss' modifying crudely
                    hrsminssecs.insert(0,'00')
                out.append(sum([a*b for a,b in zip(ftr, [int(i) for i in hrsminssecs])]))
            except:
                out.append(None)
        return out
    
    def convert_PR_name(self,name_strings : npt.ArrayLike) -> npt.ArrayLike:        
        
        out = []
        for name_string in name_strings:
            name = match('\D+(?=[0-9]+)',name_string) # Don't need start of string signifier because match goes from start of string
            if not name:
                out.append('')
                continue
            name = name.group()
            #Ensure title case as case sensitive
            name.title()
            out.append(name) 
        return out
    
    def convert_PR_categories(self,category_strings : npt.ArrayLike) -> npt.ArrayLike:
        #Will split the text here so we can just pass the full-string into the function
        #All categories seem to have a length of 7
        
        out = []
        for category_string in category_strings:
            if type(category_string) is not str:
                out.append('')
                continue
            category_string = category_string[0:7]
                
            
            #CASE remove whitespaces at any point in the string
            category_string = sub('\s','',category_string)
            
            #CASE W in stead of F (Ladies, not Female)
            category_string = sub('W|w','F',category_string)
            
            #CASE put anyone in intermediate categors (M45) into the main category (M40)
            #Tried to expand to include things like M45-49 of M40-44 to basically replace all with a 0
            # This is excessive as will always match M40 and replace with 0 needlessly
            category_string = sub('(?<=[0-9])(0|5)\S*','0',category_string)
            
            #At this point, any age category that ends in 0 or 5 should be in the form - SM40, SF30 VM50 etc.
            
            #Remove Vs
            #CASE remove any V instances for Vet (V or Vet)
            category_string = sub('(?i)v(et){0,1}','',category_string)
            
            #Remove the Ss too, modify to SENIOR afterwards
            category_string = sub('(?i)s','',category_string)
            
            # Any category with a number under 30 will be put into SENIOR if over 18 but U18 if under 18
            #Get last number in string
            number = search('[0-9]+$',category_string)
            if not number:
                out.append('')
                continue
            number = int(number.group())
            if number < 30:
                if number > 18:
                    #Check for match with M
                    if search('(m|M)',category_string):
                        category_string = 'MSENIOR'
                    else:
                        category_string = 'FSENIOR'
                
                else:
                    #Check for match with M
                    if search('(m|M)',category_string):
                        category_string = 'MU18'
                    else:
                        category_string = 'FU18'
            
            
            
            out.append(str.upper(category_string))
        return out

def convert_categories(category_strings : npt.ArrayLike) -> npt.ArrayLike:
    """Converts the race categories by trying to deal with the variety of ways categories are represented
    For example it may replace 'Open' with 'Senior' we are assuming it is not truly open and people of specific age will be in their specific categories
    Will add more examples with time

    Args:
        category_strings (npt.NDArray): The string of categories from a results table

    Returns:
        npt.NDArray: the cleaned and changed strings so they should match those in the database
    """
    from re import sub, match
    
    out = []
    
    for category_string in category_strings:
        
        #CASE L\W in stead of F (Ladies, not Female)
        category_string = sub('L|l|W|w','F',category_string)
        #CASE remove whitespaces at any point in the string
        category_string = sub('\s','',category_string)
        #CASE remove J from the string
        category_string = sub('(?i)j','',category_string)
        
        #CASE M or F only, append SENIOR
        if match('(M|F)$',category_string):
            category_string = category_string + 'SENIOR'
        
        category_string = sub('(?i)sen$','SENIOR',category_string)
        #CASE replace SENR with SENIOR, decided easier to do individual cases than try one shaky catch all
        
        #CASE replace OPEN with SENIOR
        category_string = sub('(?i)open','SENIOR',category_string)
        #CASE replace SEN with SENIOR

        category_string = sub('(?i)senr$','SENIOR',category_string)
        #Eradicate any U22,U23 category should it exist, SENIOR NOW!!
        category_string = sub('(?i)u*2[2+3]','SENIOR',category_string)
        
        #Add a U to the U18 category if it doesn't have one
        category_string = sub('(?i)u*1[7,8]','U18',category_string)
        
        #Add a U to the U21 category if it doesn't have one
        category_string = sub('(?i)u*21','U21',category_string)
        
        #There may be a loose U now if U was before a sex qualifier (UM23 for example)
        category_string = sub('(?i)u(?![0-9])','',category_string)
        
        #CASE put anyone in intermediate categors (M45) into the main category (M40)
        #Tried to expand to include things like M45-49 of M40-44 to basically replace all with a 0
        # This is excessive as will always match M40 and replace with 0 needlessly
        category_string = sub('(?<=[0-9])(0|5)\S*','0',category_string)
        #CASE remove any V instances for Vet (V or Vet)
        category_string = sub('(?i)v(et){0,1}','',category_string)
        # Capitalise the string before returning so it matches the database
        out.append(str.upper(category_string))
        
    return out

def clean_position_date(position_values : npt.ArrayLike) -> npt.ArrayLike:
    out=[]
    if position_values.dtype == 'int64' or position_values.dtype == 'float64':
        return position_values # No need to transform!
    from re import sub,match
    #Remove leading or trailing spaces
    for position_value in position_values:        
        #Check that there aren't any text values in a string
        position_value = position_value.strip()
        #Remove any whitespaces
        position_value = sub('\s','',position_value)
        #Take the numbers from the start of the sequence only
        numbers_loc =  match('[0-9]+',position_value)
        if numbers_loc == None:
            out.append(None)
        else:        
            out.append(int(numbers_loc.group()))
        
    return out