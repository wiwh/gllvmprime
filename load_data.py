import pandas as pd
import numpy as np
from pathlib import Path

# This file merges all the datasets into a 3D array (individual, variable, time), 
# with 12 variables

# Variable names, appearing in this order:

# Birthyear
# Gender
# Studies
# Date Month (Binary)	JD151
# Date Day (Binary)	JD152
# Date Year (Binary)	JD153
# Date Day/Week (Binary)	JD154
# Cut Paper (Binary)	JD155
# Cactus (Binary)	JD156
# President (Binary)	JD157
# Vice president (Binary)	JD158
# Series 7 (count 0-5)	JD142-JD143-JD144-JD145-JD146
# Count Backwards (ordinal 0-2)	JD120-JD125
# WORDS Recall Immediate (count 0-10)	JD174
# WORDS Delayed (count 0-10): RD184	JD184

# In addition, the index is defined by the following variables:
# * HHID: household identification number
# * PN: person identifier number (sub-index)


# Read the variable names from the CSV fil

# Set the paths for data files, and initial setup 
# -----------------------------------------------
data_path = Path("data/HRS/data")
varnames_path = Path("data/HRS/variable_names.csv")
save_path = Path("data/HRS/HRS_3Darray.npy")
years_series = list(range(2004, 2021, 2))

# Create a variable_location dictionary
# ------------------------------------
# This stores the loc of the columns for each of the variables
variable_location={
    'birthyear':0,
    'gender':1,
    'education':2,
    'month':3,
    'day':4,
    'year':5,
    'day_week':6,
    'paper':7,
    'cactus':8,
    'president':9,
    'vice_president':10,
    'serie_7':list(range(11,16)),
    'backwards':list(range(16, 18)),
    'recall_immediate':18,
    'recall_delayed':19
    }


#%% Load and merge all data into an array
# ---------------------------------------

# Get the list of CSV filenames in the data path
csv_filenames = list(data_path.glob("*.csv"))

# Read each CSV file into separate DataFrames and store them in a list
csv_files = [pd.read_csv(csv_filename, na_values=' ') for csv_filename in csv_filenames]

# Define the years and create names for the dictionary
years = [f'{year}' for year in years_series]
names = years + ["tracker"]

# Create a dictionary with the CSV file DataFrames, using the names as keys
data = {key:values for key, values in zip(names, csv_files)}

# Extract the 'tracker' DataFrame and set 'HHID' and 'PN' as the index
tracker = data.pop("tracker").set_index(['HHID', 'PN'])
tracker = tracker[['BIRTHYR', "GENDER", "DEGREE"]]


varnames = pd.read_csv(varnames_path)


# Create a new dictionary 'data_new' to store selected variables from each DataFrame
data_new = {}
for y, df in data.items():
    # Select variables based on 'varnames[y]' and set 'HHID' and 'PN' as the index
    data_new[y] = data[y][varnames[y].to_list()]
    data_new[y].set_index(['HHID', 'PN'], inplace=True)
    # Provide a unique name for tracker variables for this year (even though they are duplicated, this is necessary for a proper merge later on)
    tracker.rename(columns=dict(zip(tracker.columns, tracker.columns+y)), inplace=True)
    # Merge 'tracker' and selected variables based on the common index
    data_new[y] = pd.merge(tracker, data_new[y], left_index=True, right_index=True, how="right")


# Check that no observations were lost during the merging process
nobs_origin = [len(data) for _, data in data.items()]
nobs_final  = [len(data) for _, data in data_new.items()]
assert(nobs_origin == nobs_final)

# Merge all the DataFrames in 'data_new' together, using how="outer" to keep the maximum information
merged_df = list(data_new.values())[0]
for _, df in list(data_new.items())[1:]:
    merged_df = pd.merge(merged_df, df, left_index=True, right_index=True, how="outer")

# Create the 3D tensor
myarray = np.array(np.split(merged_df.values, 9, axis=1)).transpose(1,2,0)
array_final = myarray.copy()
nan_mask = np.isnan(myarray)


# Create the final dataset as dictionary of arrays, which we then concatenate
# ---------------------------------------------------------
data = {}

#%% Age 
# ------------------------------------------------------------
# Use the column 0, birthyear, and transform it to age
# Average the NA out to obtain the birthyear
birthyears = np.round(np.nanmean(myarray[:,variable_location['birthyear']], axis=1))
birthyears = birthyears[:,np.newaxis, np.newaxis]
birthyears = np.repeat(birthyears, len(years_series), axis=2)
years = np.array(years_series)[np.newaxis, np.newaxis,:]
data['age'] = years - birthyears
data['age']



#%% Gender
#---------
# Coded as 1=Male, 2=Female, and transform it to 0=Male, 1=Female
# Average the NA out to obtain the gender
gender = np.round(np.nanmean(myarray[:,variable_location['gender']], axis=1))
gender[gender==1] = 0
gender[gender==2] = 1
gender = gender[:,np.newaxis, np.newaxis]
gender = np.repeat(gender, len(years_series), axis=2)
data['gender'] = gender

#%% Education
# ---------------
# Education has the following encoding:
#  0.  No degree
#  1.  GED
#  2.  High school diploma
#  3.  Two year college degree
#  4.  Four year college degree
#  5.  Master degree
#  6.  Professional degree (Ph.D., M.D., J.D.)
#  9.  Degree unknown/Some College


# We will replace 9. by nan
# Average the NA out
education = np.round(np.nanmean(myarray[:,variable_location['education']], axis=1)) 
education = education[:, np.newaxis, np.newaxis]
education[education==9] = np.nan
education = np.repeat(education, len(years_series), axis=2)
data['education'] = education

#%% Transform the recollection questions (OK/not OK), which are all coded as:
#      1.  ANSWER OK
#      5.  ANSWER NOT OK
#      8.  DK (Don't Know); NA (Not Ascertained)
#      9.  RF (Refused)
#  Blank.  INAP (Inapplicable); Partial Interview
# We encoded as 1 if answer is ok (=1), 0 otherwise, and NA stay nA.

recollection_list = ['month', 'day', 'year', 'day_week', 'paper', 'cactus', 'president', 'vice_president']

for recollection_item in recollection_list:
    # Create a View
    recollection_data = myarray[:,variable_location[recollection_item]]
    # Transform
    recollection_data[recollection_data != 1] = 0
    # Put back NA
    recollection_data[nan_mask[:, variable_location[recollection_item]]] = np.nan
    # Change the axis
    recollection_data = recollection_data[:,np.newaxis,:]
    # Store in dictionary
    data[recollection_item] = recollection_data

#%% Serie_7
# -------
# The serie_7 vairable sums the correct answers of all 5 "substract 7 from previous number" questions

for i in range(len(variable_location['serie_7'])):
    correct_answer = 100 - (i+1)*7
    myarray[:,variable_location['serie_7'][i],:] = \
        myarray[:,variable_location['serie_7'][i],:] == correct_answer 
myarray[nan_mask] = np.nan

# If all are nan, return nan, else return the sum of correct answers
serie_7_nan = nan_mask[:, variable_location['serie_7'], :].sum(axis=1, keepdims=True) == 5
serie_7 = np.nansum(myarray[:,variable_location['serie_7'],:], axis=1, keepdims=True)
serie_7[serie_7_nan] = np.nan

data['serie_7'] = serie_7

#%% Backwards
#------------
#  Create a new variable that summarizes the two "count backwards" tasks.
backwards_nan = nan_mask[:, variable_location['backwards']].sum(axis=1, keepdims=True) == 2
backwards = (myarray[:, variable_location['backwards'][0], :] == 1)  * 1. + (myarray[:, variable_location['backwards'][0],:] != 1) * (myarray[:,variable_location['backwards'][1],:] == 1) * 2.
backwards = backwards[:, np.newaxis,:]
backwards[backwards_nan] = np.nan

#%% Recall_immediate
#-------------------
# The data has already been processed
data['recall_immediate'] = myarray[:,variable_location['recall_immediate']][:,np.newaxis,:]

#%% Recall_delayed
#-----------------
# The data has already been processed
data['recall_delayed'] = myarray[:,variable_location['recall_delayed']][:,np.newaxis,:]


#%% Create the final array and save
# -------------------------------
data_final = np.concatenate([arr for arr in data.values()], axis=1)
np.save(save_path, myarray)
