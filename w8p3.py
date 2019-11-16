
# coding: utf-8

# # Week 8 Problem 3
# 
# If you are not using the `Assignments` tab on the course JupyterHub server to read this notebook, read [Activating the assignments tab](https://github.com/lcdm-uiuc/info490-sp17/blob/master/help/act_assign_tab.md).
# 
# A few things you should keep in mind when working on assignments:
# 
# 1. Make sure you fill in any place that says `YOUR CODE HERE`. Do **not** write your answer in anywhere else other than where it says `YOUR CODE HERE`. Anything you write anywhere else will be removed or overwritten by the autograder.
# 
# 2. Before you submit your assignment, make sure everything runs as expected. Go to menubar, select _Kernel_, and restart the kernel and run all cells (_Restart & Run all_).
# 
# 3. Do not change the title (i.e. file name) of this notebook.
# 
# 4. Make sure that you save your work (in the menubar, select _File_  → _Save and CheckPoint_)
# 
# 5. You are allowed to submit an assignment multiple times, but only the most recent submission will be graded.
# -----
# # Problem 8.3. Web Scraping
# In this problem, we are going to extract information from the website of [World Health Organization](http://www.who.int/countries/en/).

# In[1]:

from IPython.display import HTML, display
import requests
import bs4
from bs4 import BeautifulSoup
import re
from lxml import html
import numpy as np
import pandas as pd

from nose.tools import assert_equal, assert_is_instance
from numpy.testing import assert_array_equal
from pandas.util.testing import assert_frame_equal


# Here is the link to the website that we are going to explore:

# In[2]:

who = 'http://www.who.int/countries/en/'


# ## 1. Function: get_country_url
# Write a function named `get_country_url` that takes a country name and the WHO url, and returns the corresponding url link (string) to the webpage of that country. For example, if the input country name is `"France"`, then you should return `"http://www.who.int/countries/fra/en/"`. You may want to inspect the elements of the website on your browser.

# In[3]:

def get_country_url(country, url=who):
    '''
    Finds the url link of the input country on the WHO website.
    
    Parameters
    ----------
    country: A string. Name of the country.
    url: A string. Default: 'http://www.who.int/countries/en/'
    
    Returns
    -------
    A string. Url link of the country.
    '''
    #Get url for a country
    joey_bag_of_donuts=requests.get(url).content
    the_homewrecker=BeautifulSoup(joey_bag_of_donuts,'lxml')
    art_vandalay = the_homewrecker.find(string=country)
    joey_jr=art_vandalay.find_parent('a', href=True)
    country_url='http://www.who.int'+joey_jr['href']
        
    return country_url


# In[4]:

t1_url = get_country_url('Panama')
assert_equal(t1_url, 'http://www.who.int/countries/pan/en')

t2_url = get_country_url('United Kingdom')
assert_equal(t2_url, 'http://www.who.int/countries/gbr/en')

t3_url = get_country_url('Micronesia (Federated States of)')
assert_equal(t3_url, 'http://www.who.int/countries/fsm/en')


# ## 2. Function: get_country_stats
# 
# If you click the link for a country, `France` for example, you are going to see the following table: 
# ![](./images/france.png)
# Now we'd like to get the data from that table.
# 
# In the following code cell, write a function named `get_country_stats` that takes a country name and the WHO url, and returns a 2d numpy array containing rows of the HTML table. In the `France` example, the output should look like this:
# ```
# [['Total population (2015)', '64,395,000'],
#  ['Gross national income per capita (PPP international $, 2013)', '37'],
#  ['Life expectancy at birth m/f (years, 2015)', '79/85'],
#  ['Probability of dying under five (per 1 000 live births, 0)',
#   'not available'],
#  ['Probability of dying between 15 and 60 years m/f (per 1 000 population, 2015)',
#   '104/51'],
#  ['Total expenditure on health per capita (Intl $, 2014)', '4,508'],
#  ['Total expenditure on health as % of GDP (2014)', '11.5']]
# ```

# In[7]:

def get_country_stats(country, url=who):
    '''
    Finds the statistical data of the input country on the country's website.
    
    Parameters
    ----------
    country: A string. Name of the country.
    url: A string. Default: 'http://www.who.int/countries/en/'
    
    Returns
    -------
    A 2d numpy array of identical content as the table on the website of the country.
    '''
    #Get statistical data for the input country
    homewrecker_jr=get_country_url(country,url)
    art_vandalay_jr=requests.get(homewrecker_jr).content
    overachiever=BeautifulSoup(art_vandalay_jr,'lxml')
    funk_meister=overachiever.find('table',class_='tableData').tbody
    unanimous_decision=[row for row in funk_meister.find_all('tr')]
    stats=[]
    for i in unanimous_decision:
        taco_called_wanda=[]
        for j in i:
            if j.string=='\n':
                continue
            if j.string==None:
                j.string='not available'
            taco_called_wanda.append(j.string)
        stats.append(taco_called_wanda)    
    
    return stats


# In[8]:

# print out the result for France
t1_stats = get_country_stats('France')
for col, num in t1_stats:
    print('{0:80s}: {1:s}'.format(col, num))


# In[9]:

france = [['Total population (2015)', '64,395,000'],
          ['Gross national income per capita (PPP international $, 2013)', '37'],
          ['Life expectancy at birth m/f (years, 2015)', '79/85'],
          ['Probability of dying under five (per 1 000 live births, 0)', 'not available'],
          ['Probability of dying between 15 and 60 years m/f (per 1 000 population, 2015)', '104/51'],
          ['Total expenditure on health per capita (Intl $, 2014)', '4,508'],
          ['Total expenditure on health as % of GDP (2014)', '11.5']]
assert_array_equal(t1_stats, france)

germany = [['Total population (2015)', '80,688,000'],
           ['Gross national income per capita (PPP international $, 2013)', '44'],
           ['Life expectancy at birth m/f (years, 2015)', '79/83'],
           ['Probability of dying under five (per 1 000 live births, 0)', 'not available'],
           ['Probability of dying between 15 and 60 years m/f (per 1 000 population, 2015)', '87/47'],
           ['Total expenditure on health per capita (Intl $, 2014)', '5,182'],
           ['Total expenditure on health as % of GDP (2014)', '11.3']]
t2_stats = get_country_stats('Germany')
assert_array_equal(t2_stats, germany)

andorra = [['Total population (2015)', '70,000'],
           ['Gross national income per capita (PPP international $, 0)', 'not available'],
           ['Life expectancy at birth m/f (years, 0)', 'not available'],
           ['Probability of dying under five (per 1 000 live births, 0)', 'not available'],
           ['Probability of dying between 15 and 60 years m/f (per 1 000 population, 0)', 'not available'],
           ['Total expenditure on health per capita (Intl $, 2014)', '4,273'],
           ['Total expenditure on health as % of GDP (2014)', '8.1']]
t3_stats = get_country_stats('Andorra')
assert_array_equal(t3_stats, andorra)


# ## 3. Function: get_all_countries
# 
# On the WHO webpage, there are 194 member states. Write a function named `get_all_countries` that takes the WHO url as input, and returns a list of all 194 country names (a list of strings). The order of country names should be identical to the order on the WHO webpage, i.e. alphabetical order. *Hint:* find tags and attributes that makes country names distinct from other website elements.

# In[10]:

def get_all_countries(url=who):
    '''
    Finds names of 194 memeber states on the WHO webpage as a list of strings.
    
    Parameters
    ----------
    url: A string. Default: 'http://www.who.int/countries/en/'
    
    Returns
    -------
    A list of the names of 194 WHO member states.
    '''
    #get all of the countries
    chicken_club=requests.get(url).content
    john_coctostan=BeautifulSoup(chicken_club,'lxml')
    super_kingpin=john_coctostan.find_all('ul',class_='index')
    countries=[]
    for i in super_kingpin:
        billy_barou=i.find_all('span')
        for j in billy_barou:
            countries.append(j.text)
    
    return countries


# In[11]:

country_names = get_all_countries()
answer = ['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua and Barbuda', 'Argentina', 'Armenia', 
          'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 
          'Belize', 'Benin', 'Bhutan', 'Bolivia (Plurinational State of)', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 
          'Brunei Darussalam', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon', 'Canada', 
          'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo', 'Cook Islands', 'Costa Rica', 
          "Côte d'Ivoire", 'Croatia', 'Cuba', 'Cyprus', 'Czechia', "Democratic People's Republic of Korea", 
          'Democratic Republic of the Congo', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 
          'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia', 
          'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 
          'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran (Islamic Republic of)', 'Iraq', 'Ireland', 'Israel', 
          'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Kuwait', 'Kyrgyzstan', 
          "Lao People's Democratic Republic", 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Lithuania', 'Luxembourg', 
          'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 
          'Mexico', 'Micronesia (Federated States of)', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 
          'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Niue', 'Norway', 'Oman', 
          'Pakistan', 'Palau', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 
          'Republic of Korea', 'Republic of Moldova', 'Romania', 'Russian Federation', 'Rwanda', 'Saint Kitts and Nevis', 
          'Saint Lucia', 'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 
          'Senegal', 'Serbia ', 'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 
          'South Africa', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 
          'Syrian Arab Republic', 'Tajikistan', 'Thailand', 'The former Yugoslav Republic of Macedonia', 'Timor-Leste', 'Togo', 
          'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine', 
          'United Arab Emirates', 'United Kingdom', 'United Republic of Tanzania', 'United States of America', 'Uruguay', 
          'Uzbekistan', 'Vanuatu', 'Venezuela (Bolivarian Republic of)', 'Viet Nam', 'Yemen', 'Zambia', 'Zimbabwe']
assert_array_equal(answer, country_names)


# ## 4. Function: get_combined_dataframe
# 
# Write a funciton named `get_combined_dataframe` that takes a list of country names and a list of columns as input, and returns a pandas DataFrame constructed from the statistical data found on the website of each country. Use the input list of columns as column names of the DataFrame, and the input list of country names as index. We'll use the following list of columns named `cols` as default for the function:

# In[12]:

cols = ['Total population',
        'Gross national income per capita (PPP international $)',
        'Life expectancy at birth m/f (years)',
        'Probability of dying under five (per 1 000 live births)',
        'Probability of dying between 15 and 60 years m/f (per 1 000 population)',
        'Total expenditure on health per capita (Intl $)',
        'Total expenditure on health as % of GDP']


# The reason why we need to specify the column names is that the data for some countries are obtained in different years as the rest of countries. For example, `Gross national income per capita` for `Bahamas` is taken in year 2012, whereas that for most other countries is taken in year 2013, which makes its table have different data names. Other than the year differece, data for all countries have identical structure, so I remove the years in the column names to simplify the task.
#  
# If the input is `['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola']`, then your output should look like this: 
# ![](./images/table.png)
# To keep data identical to what you extract from each country's website, make sure the data type of the DataFrame is string for all columns.
# 
# Note that the result from `get_country_stats` is a column-oriented list (each element in the list represents a column to put in the DataFrame). If you wish to operate with a row-oriented list, you can always convert it by using `np.transpose`.

# In[13]:

def get_combined_dataframe(countries, cols=cols):
    '''
    Combines data for each country as a dataframe using specified column names as columns and country names as index.
    
    Parameters
    ----------
    country: A list of string. Names of the countries.
    cols: A list of string. Default: the list defined above this cell.
    
    Returns
    -------
    A pandas DataFrame object.
    '''
    #Create dataframe from d
    close_taker=[]
    for c in countries:
        personal_trainer=np.transpose(get_country_stats(c))
        close_taker.append(personal_trainer[1])
    combined_df=pd.DataFrame(close_taker,index=countries,columns=cols)    
    
    return combined_df


# In[14]:

countries1 = ['China', 'Egypt', 'United States of America']
df1 = get_combined_dataframe(countries1)
assert_is_instance(df1, pd.DataFrame)
a1 = pd.DataFrame ([['1,400,000,000', '11', '75/78', 'not available', '98/71', '731','5.5'],
                    ['91,508,000', '10', '69/73', 'not available', '196/119', '594','5.6'],
                    ['321,774,000', '53', '77/82', 'not available', '128/77', '9,403','17.1']], 
                   columns=cols, index=countries1)
assert_frame_equal(df1, a1)

countries2 = country_names[100:110]
df2 = get_combined_dataframe(countries2)
a2 = pd.DataFrame([['24,235,000', '1', '64/67', 'not available', '245/196', '44', '3.0'],
                   ['17,215,000', '750', '57/60', 'not available', '398/330', '93','11.4'],
                   ['30,331,000', '22', '73/77', 'not available', '167/79', '1,040','4.2'],
                   ['364,000', '9', '77/80', 'not available', '79/43', '1,996', '13.7'],
                   ['17,600,000', '1', '58/58', 'not available', '266/267', '108','6.9'],
                   ['419,000', '28', '80/84', 'not available', '70/37', '3,072', '9.8'],
                   ['53,000', '4', 'not available', 'not available', 'not available','680', '17.1'],
                   ['4,068,000', '2', '62/65', 'not available', '227/182', '148', '3.8'],
                   ['1,273,000', '17', '71/78', 'not available', '190/99', '896', '4.8'],
                   ['127,017,000', '16', '74/80', 'not available', '161/82', '1,122','6.3']],
                 columns=cols, index=countries2)
assert_frame_equal(df2, a2)


# In[ ]:



