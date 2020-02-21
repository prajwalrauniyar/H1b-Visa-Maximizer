##Created by Prajwal Rauniyar

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import pickle

prefix = 'h1b_datahubexport-'
first_year = 2009
num_years = 11
num_features = 5 

dir_path = '/Users/prajwalrauniyar/Academics/Analatical_Data_Mining/Project'

init_read_pd = []
for i in range(first_year, first_year+num_years):
    a_pd = pd.read_csv( os.path.sep.join([dir_path, prefix+str(i)+'.csv']) , low_memory=False, thousands=',')
    a_pd['ZIP'] = a_pd['ZIP'].astype(float).fillna(value='None').astype(str)
    a_pd['City'] = a_pd['City'].fillna(value='None')
    a_pd['Employer'] = a_pd['Employer'].fillna(value='None')
    a_pd['Tax ID'] = a_pd['Tax ID'].fillna(value=a_pd['Tax ID'].median())
    key = a_pd[['Employer','City', 'ZIP']]
    a_pd['Employer'] = pd.Series(map(lambda x: '___'.join(x), key.values.tolist()))
    
    a_pd = a_pd.drop(columns=['City','ZIP'])
    total_sum = a_pd.to_numpy()[:,2:6].sum(axis=1)
    total_deny = a_pd.to_numpy()[:,3] + a_pd.to_numpy()[:,5]
    
    a_pd.insert(2, "Total_Filed", total_sum, True)
    a_pd.insert(3, "Total_Denied", total_deny, True)
    
    #init_read_pd.append( a_pd )
    #continue

    a_pd = a_pd.drop(columns=['Initial Approvals','Initial Denials', 
                              'Continuing Approvals','Continuing Denials',
                              'Fiscal Year','State'])
    
    init_read_pd.append( a_pd )

#Dictionary to map Industry Code to Type
#Source https://www.census.gov/cgi-bin/sssd/naics/naicsrch?chart=2007
inds_to_type = {
11	: 'Agriculture, Forestry, \nFishing and Hunting',
21	: 'Mining, Quarrying, and \nOil and Gas Extraction',
22	: 'Utilities',
23	: 'Construction',
31  : 'Manufacturing',
32  : 'Manufacturing',
33	: 'Manufacturing',
42	: 'Wholesale Trade',
44	: 'Retail Trade',
45	: 'Retail Trade',
48	: 'Transportation and Warehousing',
49	: 'Transportation and Warehousing',
51	: 'Information',
52	: 'Finance and Insurance',
53	: 'Real Estate and Rental and Leasing',
54	: 'Professional, Scientific, and \nTechnical Services',
55	: 'Management of Companies and \nEnterprises',
56	: 'Administrative and Support \nand \n Waste Management and \nRemediation Services',
61	: 'Educational Services',
62	: 'Health Care and Social \nAssistance',
71	: 'Arts, Entertainment, and \nRecreation',
72	: 'Accommodation and Food Services',
81	: 'Other Services \n(except Public Administration)',
92	: 'Public Administration'
}
#######################################    
#PLOTTING ALL THE SUM OF THE FEATURES FOR ALL YEARS
plt.figure(figsize=(20,15))

for yr in range(0,num_years):
  Y = np.sort( init_read_pd[yr].to_numpy()[:,1] )
  x_axis = np.arange(len(Y))
  plt.plot(x_axis,Y, label='%s' %(yr+first_year))

plt.legend(loc='upper left')
plt.show()

#######################################
#Plot total filed
plt.clf()
Y = []
for yr in range(0,num_years):
  _X = init_read_pd[yr].to_numpy()
  Y.append(_X[:,1].sum())
x_axis = np.arange(len(Y)) + first_year
plt.figure(figsize=(20,15))
plt.plot(x_axis,Y)
plt.xlabel('Year')
plt.ylabel('Num Visas Filed')
plt.show()

#######################################
#Plot of approved and filed
plt.clf()
Y = []
Y2 = []
for yr in range(0,num_years):
  _X = init_read_pd[yr].to_numpy()
  Y.append( (_X[:,1] - _X[:,2] ).sum())
  Y2.append( _X[:,1].sum() )

x_axis = np.arange(len(Y)) + first_year
plt.figure(figsize=(20,15))

plt.plot(x_axis,Y,label='Approvals')
plt.plot(x_axis,Y2,label='Total Filed')

plt.legend(loc='upper left')

plt.xlabel('Year')
plt.ylabel('Num Visas')
plt.show()

#######################################
#To find number of employers willing to file H-1B
plt.clf()
plt.figure(figsize=(20,15))
Y = []
for yr in range(0,11):
  _X = init_read_pd[yr].to_numpy()
  Y.append( _X.shape[0] )

plt.plot(x_axis,Y)
plt.xlabel('Year')
plt.ylabel('Num willing to file H-1B')
plt.show()


#######################################


##################################################
#Run K means with k=3 and get employers who file
#alot of visas

#BY GROUPING THEM WE GET THE EMPLOYERS THAT HAVE FILED
#VERY LESS INTO A FEW GROUPS AND THE ONES THAT HAVE FILED 
#MUCH MUCH MORE 
per_year_clean = []
for yr, a_pd in enumerate(init_read_pd):
  _X = a_pd.to_numpy()
  if yr != 0:
    kmns = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=392)
    kmns.fit(_X[:,1:3])
    _Y = _X[kmns.labels_ == 1]
    _Y = np.append(_Y, _X[kmns.labels_ == 2], axis=0)
    _Y = np.append(_Y, _X[kmns.labels_ == 3], axis=0)
    _Y = np.append(_Y, _X[kmns.labels_ == 4], axis=0)
  else:
    kmns = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=392)
    kmns.fit(_X[:,1:3])
    _Y = _X[kmns.labels_ == 1]
    _Y = np.append(_Y, _X[kmns.labels_ == 2], axis=0)
  per_year_clean.append(_Y)
  
#Get the number of unique employers
seen_emp = np.array([])
for a_np in per_year_clean:
    seen_emp = np.unique( np.append(a_np[:,0], seen_emp) )
    
#####################################################
#Year wise which industry files the most H-1B visas
#MULTIPLE graphs

for yr, clean in enumerate(per_year_clean):
  plt.clf()
  plt.figure(figsize=(18,20))
  indust_obj = []
  indust = np.unique(clean[:,3])
  y_axis = [] 
  for i in indust:
    y_axis.append( clean[clean[:,3] == i].shape[0] )
    indust_obj.append(inds_to_type[i])

  x_axis = np.arange(len(y_axis))

  plt.barh(x_axis, y_axis, align='center', alpha=0.5)
  plt.yticks(x_axis,indust_obj)
  plt.xlabel('Number of Employers')
  plt.title('YR:%s num of Employers per Industry' %str(yr+first_year))

  plt.show()

#####################################################
#Print the Top 10 employers by number of visas filed
#Also their Approval Rate for visas
for yr, clean in enumerate(per_year_clean):
  idxs = (-clean[:,1]).argsort()[:10]
  allofthem = clean[idxs]
  print('\n\nTop 10 Num of H-1B in Year:%s' %str(yr+first_year))
  for each in allofthem:
    print('\tFiled:%s\tApprovalRate:%.3f\t%s' %(each[1], 
                                        (each[1]-each[2])*100/each[1],
                                        each[0].split('__')[0]))
