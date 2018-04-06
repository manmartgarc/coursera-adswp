import pandas as pd
import numpy as np

def answer_one():
    energy = (pd.read_excel('Energy Indicators.xls',
                           skiprows = 17,
                           skip_footer = 38,
                           usecols = [2,3,4,5],
                           names = ['Country', 'Energy Supply', 
                                    'Energy Supply per Capita', '% Renewable']))

    GDP = pd.read_csv('world_bank.csv',
                      header = 4)

    ScimEn = pd.read_excel('scimagojr-3.xlsx')

    energy = energy.replace('...', np.NaN)
    energy['Country'] = energy['Country'].str.replace(r"\(.*\)","")
    energy['Country'] = energy['Country'].str.replace(r"[0-9]","")
    energy['Country'] = energy['Country'].replace({"Republic of Korea": "South Korea",
                                          "United States of America": "United States",
                                          "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
                                          "China, Hong Kong Special Administrative Region": "Hong Kong"})

    energy['Energy Supply'] = (energy['Energy Supply'] * 1000000)


    GDP['Country Name'] = GDP['Country Name'].replace({"Korea, Rep.": "South Korea",
                                                       "Iran, Islamic Rep.": "Iran",
                                                       "Hong Kong SAR, China": "Hong Kong"})

    GDP = GDP[['Country Name', '2006', '2007', '2008','2009',
              '2010', '2011', '2012', '2013', '2014','2015']]
    GDP = GDP.rename(columns = {'Country Name' : 'Country'})

    ScimEn2 = ScimEn[ScimEn['Rank'] <= 15]
    energy = energy.set_index(energy['Country'])
    GDP = GDP.set_index(GDP['Country'])
    ScimEn2 = ScimEn2.set_index(ScimEn2['Country'])

    df1 = pd.merge(ScimEn2, GDP, how = 'inner', left_index = True, right_index = True)
    df2 = pd.merge(df1, energy, how = 'left', left_index = True, right_index = True)

    df2 = df2[['Rank', 'Documents', 'Citable documents', 'Citations',
               'Self-citations', 'Citations per document', 'H index',
               'Energy Supply', 'Energy Supply per Capita', '% Renewable',
               '2006', '2007', '2008', '2009','2010','2011','2012','2013',
               '2014','2015']].sort('Rank')
    return df2
answer_one()