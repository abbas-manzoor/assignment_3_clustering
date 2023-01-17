import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_data(file_name, indicators):
    
    df = pd.read_csv(file_name, index_col=False)
    df = df[df['Indicator Name'].isin(indicators)]
    df.drop(['Country Code', 'Indicator Code', 'Indicator Name'],
                axis=1, inplace=True)
    
    country = ['Australia', 'Bangladesh', 'Germany', 
               'United Kingdom','India', 'Pakistan']
    
#     filter cities
    print(country)
    df= df[df['Country Name'].isin(country)]
    
    year_as_column = df.set_index('Country Name')
    country_as_column = year_as_column.T
    return year_as_column.reset_index(), country_as_column


# defining indicators 
indicators = ['Population growth (annual %)' ]

# reading data 
data = get_data("API_19_DS2_en_csv_v2_4773766/API_19_DS2_en_csv_v2_4773766.csv", indicators )[0]


# reseting index 
data = data.set_index("Country Name").T
data.reset_index(inplace=True)
data.rename(columns={ 'index' : 'Year' })
print(list(data.columns))


# show actual points 
country_name = list(data.columns)[1:]
plt.plot(data['index'],data[country_name[4]])
# plt.title(data[country_name[4]])
fig = plt.figure()
spacing = 0.50
fig.subplots_adjust(bottom=spacing)
plt.show()

# normalisong datasets
undersample_data = data.loc[np.linspace(data.index.min(),
                                data.index.max(),1500).astype(int)]
undersample_data = undersample_data.reset_index().drop('index',axis=1)

# print after normalize
plt.plot(undersample_data.level_0,undersample_data['Australia'])
plt.xlabel('Year',fontsize=20)
plt.ylabel('Population Growth',fontsize=20)

# fit data into model 
data_array = np.array(undersample_data.T.drop('level_0').values)
from tslearn.clustering import TimeSeriesKMeans
model = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=10)
model.fit(data_array)
cities_list = undersample_data.T.drop('level_0').index.tolist()


y=model.predict(data_array)
x = undersample_data.level_0

# plot the scatter plot 
plt.figure(figsize=(20,20))
k_dict = {'1':0,'2':0,'3':0,'4':1,'5':1,'6':1,'7':2,'8':2,'9':2}
colors = ['navy']*3+['darkorange']*3+['k']*3
Names = ['Class 0']*3+['Class 1']*3+['Class 2']*3
for j in range(1,10):
    plt.subplot(3,3,j)
    k = np.random.choice(np.where(y==k_dict[str(j)])[0])
    plt.plot(x,data_array[k],'.',color=colors[j-1])
    plt.ylabel('Population Growth',fontsize=20)
    plt.xlabel('Year',fontsize=20)
    plt.title('City=%s, Class = %s'%(cities_list[k],Names[j-1]),fontsize=20)
    plt.ylim(data_array.min(),data_array.max())


# plot the histogram for clear understanding 
plt.figure(figsize=(20,20))
k_dict = {'1':0,'2':0,'3':0,'4':1,'5':1,'6':1,'7':2,'8':2,'9':2}
colors = ['navy']*3+['darkorange']*3+['k']*3
Names = ['Class 0']*3+['Class 1']*3+['Class 2']*3
for j in range(1,10):
    plt.subplot(3,3,j)
    k = np.random.choice(np.where(y==k_dict[str(j)])[0])
    plt.hist(data_array[k],color=colors[j-1])
    plt.ylabel('Population Growth',fontsize=20)
    plt.xlabel('Year',fontsize=20)
    plt.title('City=%s, Class = %s'%(cities_list[k],Names[j-1]),fontsize=20)
    plt.xlim(data_array.min(),data_array.max())
