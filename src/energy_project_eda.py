"""
This code is meant for the exploratory data analysis of the New York Buildings Energy data
DOMAIN KNOWLEDGE

        - Building owners make self reporting of the needed information required by law
        - The Score is calculated by the New York Environmental Protection Agency's ENERGY STAR Portfolio Manager
        - The law applies to taxable lots with a gross floor area of 50,000 square feet and above 
        - For a collection of taxable lots on same land with gross floor area of 100,000 square feet and above
        - From 2018 taxable lots with floor area of 25,000 square feet are required to make this self declaration


The ENERGY STAR® Energy Performance Scorecard
This scorecard provides a quick snapshot of a building’s energy performance. Depending on type of building, the score
 card will display either the building’s 1-100 ENERGY STAR score or site energy use intensity (EUI).

About the 1-100 ENERGY STAR score
Using the 1 – 100 ENERGY STAR score, you can understand how a building’s energy consumption measures up against 
similar buildings nationwide. A score of 50 represents median energy performance, which means it performs better 
than 50 percent of its peers. Higher scores mean better energy efficiency, resulting in less energy used and fewer 
greenhouse gas emissions. The pin color indicates the building’s overall energy performance as follows:

Green: ENERGY STAR score = 75 – 100
Yellow: ENERGY STAR score = 26 – 74
Red: ENERGY STAR score = 0 – 25

That means it has a score of 75 or better and has been verified to perform among the top 25 percent of similar 
buildings nationwide. On average, ENERGY STAR certified buildings use 35 percent less energy and generate 35 percent 
fewer greenhouse gas emissions than their peers. 

Scores and certification are available for more than 20 different U.S. building types. Learn more about how the 
1-100 ENERGY STAR score is calculated.

About Energy Use Intensity
If a 1-100 score for a specific building type is not available, site energy use intensity (EUI) is displayed on the 
scorecard. EUI is expressed as energy (kBtu) per square foot per year.
It’s calculated by dividing the total annual energy consumed by the building by the total gross floor area of the 
building. Generally, a low EUI signifies good energy performance.
However, certain property types will always use more energy than others. For example, 
an elementary school uses relatively little energy compared to a hospital.

"""

#Load relevant modules

import pandas as pd    
import numpy as np          
import seaborn as sns
import matplotlib.pyplot as plt  
import os 
import datetime
import re


%matplotlib inline   
plt.style.use('ggplot')

pd.set_option('display.max_columns',70)

#Import the data
filename = 'nyc_energy.csv'
data = pd.read_csv(filename)

print(f'The are {data.shape[0]} rows and {data.shape[1]} columns in the data set')
data.head()
data.info()

#In the data all missing values have been hard coded as not available. We therfore replace them with NAN
data = data.replace({"Not Available":np.nan})


#It seems all columns have been imported as string so we ensure that the numeric columns as converted back 
text_in_name = ['ft²', 'kBtu', 'Metric Tons CO2e', 'kWh', 'therms', 'gal', 'Score']

for col in list(data.columns):
    if any(subtxt in col for subtxt in text_in_name):
        data[col] = data[col].astype(float)

#Let's tidy up the column names     
data.columns=data.columns.str.replace('-|\(|/|\)|,|\?',"").str.strip().str.lower().str.replace(' ','_').str.replace('__','_')

#Having read the documentation on the data set, I decided that the following may not be 
# useful for the prediction of the energy star score
features_to_drop = ['order'
                    ,'property_id'
                    ,'property_name'
                    ,'parent_property_id'
                    ,'parent_property_name'
                    ,'bbl_10_digits'
                    ,'address_2'
                    ,'nyc_borough_block_and_lot_bbl_selfreported'
                    ,'nyc_building_identification_number_bin'
                    ,'address_1_selfreported'
                    ,'postal_code'
                    ,'street_number'
                    ,'street_name'
                    ,'water_intensity_all_water_sources_galft²'
                    ,'release_date'
                    ,'water_required'
                    ,'dof_benchmarking_submission_status'
                    ,'latitude'
                    ,'longitude'
                    ,'community_board'
                    ,'council_district'
                    ,'census_tract'
                    ,'nta'
                    ,'year_built'
                    ,'list_of_all_property_use_types_at_property'
                    ,'2nd_largest_property_use_type'
                    ,'3rd_largest_property_use_type' 
                    ,'dof_gross_floor_area'
                    , 'borough'
                      ] 

# create age of the building
now = datetime.datetime.now()
data['building_age']  =   now.year - data['year_built']

# Use the first character of the reported 10 digit borough, block and lot identifier to fill in any missing in the borough 
#column
data['num_borough'] = (data['bbl_10_digits'].str[0])
borough_map = { '1' : 'Manhattan', '2' : 'Bronx', '3' :'Brooklyn', '4' :'Queens', '5': 'Staten Island'}
data['derived_borough']   = data['num_borough'].map(borough_map)

data.drop('num_borough', axis = 1, inplace = True)


# Count the number of things the propoerty is used for
f = lambda x: len(x['list_of_all_property_use_types_at_property'].split(','))  
data['number_of_property_uses'] =  data.apply(f, axis=1)


#drop the unused features
data.drop(features_to_drop, axis = 1, inplace = True)

##   MISSING VALUES AND DUPLICATE FEATURES

# number of missing values
data.isna().sum()

# percentage of missing values
round(100*data.isna().sum()/len(data),1)    

# check for duplicate features
(data.columns.duplicated())


#drop them
data = data.loc[:,~data.columns.duplicated()]

## TARGET VARIABLE EXPLORATION 

target='energy_star_score'

#How many cases of the target is missing
data[target].isna().sum()
print(f'{round(100*data[target].isna().sum()/len(data),0)} of the target is missing')



#Remove all cases where the target is missing. We need only cases where we can apply supervised learning algorithm
data.dropna(subset = [target],inplace=True)

data.shape

#summary description of target
data[target].describe()

#Histograms transformed target


fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

sns.distplot((data[target].dropna()), kde= False, ax =ax1)
sns.kdeplot(data[target].dropna(), shade = False, alpha = 0.8, ax = ax2);
#plt.xlabel('Energy Star Score', size = 20); plt.ylabel('Density', size = 20);
#plt.close(g.fig)
plt.show()


##  Determine the number of categorical and numeric features 
numeric_features=[col for col in data.columns if data[col].dtype !='O' and col not in features_to_drop + [target]]
categorical_features=[col for col in data.columns if col not in numeric_features + features_to_drop + [target]]
print(f'There are {len(numeric_features)} numeric and {len(categorical_features)} categorical featuresin the data')



## CATEGORICAL FEATURE EXPLORATION 

#what is the cardinality of each categorical feature
for col in data[categorical_features]:
    print ("----- %s ----- " % col)
    print (data[col].fillna('Missing').value_counts(dropna=False))

#export the cardinality of categorical features into an excel file: To be used in bin counting/grouping
writer = pd.ExcelWriter('./tempfiles/categorical_features_cardinality.xlsx', engine='xlsxwriter')
for col in categorical_features: 
    card_counts=data[col].fillna('Missing').value_counts(dropna = False)
    card_counts.to_excel(writer, sheet_name = col[:31])
writer.save()


# Relationship between categorical features and target
for i, col in zip(enumerate(categorical_features)):
     plt.subplot(np.ceil(len(categorical_features)/2), np.ceil(len(categorical_features)/3), i + 1)
     data.groupby(col)[target].mean().plot.barh()

fig, ax = plt.subplots(rows = np.ceil(len(categorical_features)/2), cols = np.ceil(len(categorical_features)/3))
for i, col in enumerate(data[categorical_features]):
     data.groupby(col)[target].mean().plot.barh(ax = ax[i]).set_title(col)

fig, ax = plt.subplots(3, 2)
for i, col in enumerate(categorical_features):
     data[col].fillna('Missing').value_counts().plot("bar", ax=ax[i]).set_title(col)


  fig, ax = plt.subplots(rows, cols)
        for i, feature in enumerate(df[feature_list]):
            df[feature].value_counts().plot("bar", ax=ax[i]).set_title(feature)
#####################################################################################################
##  2. NUMERIC FEATURE EXPLORATION
#####################################################################################################

# 2.1 Quantitative Description

#summary of the features
data[numeric_features].describe()

#Pairwise correlations

data[numeric_features].corr()>0.5

#correlations greater than 0.5
eda_functions.correlated_features(data,numeric_features,corr_threshold=0.6)
eda_functions.correlation_matrix_plot(data,numeric_features,corr_threshold=0.6)



#http://www.columbia.edu/~cjd11/charles_dimaggio/DIRE/styled-4/styled-11/code-8/


# 2.2 Graphical Description

# STEP 2. Outliers counts and Percentage of outliers
from scipy import statistics as stats

def zscore_outliers(df,feature_list):
    z_score = np.abs(stats.zscore(df))
    outliers = np.where(z_score > 3, 1, 0)
    ztable = pd.concat(z_score, outliers, axis=1)
    ztable.columns = ['Zscore', 'OutlierGFlag']
    return(ztable)

def IQR_Outliers(df,feature_list):
    df = df.copy()
    for col in feature_list:
        q1 = df[feature_list].quantile(0.25)
        q3 = df[feature_list].quantile(0.75)
        (df[col] < (q1 - 1.5 * (q3-q1))) |(df[col] > (q3 + 1.5 * (q3-q1)))


   
# STEP 2. Outliers counts and Percentage of outliers
# STEP 3. Summaries of numerical features
# STEP 4. Target and feature relationship
# STEP 5. Pairwise feature relationships (Correlation between pairwise features) 
# STEP 6. Constant features and Quasi constant features
# STEP 7. Categorical features cardinality counts and Categorical features values counts for each category  
# STEP 8. Categorical features and target relationships


