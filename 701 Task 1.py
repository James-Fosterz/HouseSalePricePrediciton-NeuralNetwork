import pandas as pd
import numpy as np
import datetime as dt
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from scipy.ndimage import label



#torch_tensor_output = torch.tensor(df['output'].values)
#torch_tensor_vectors = torch.from_numpy(df['vector'].values)



class initialise_data():
    def data_loader(self):      
        house_data = pd.read_csv('INM701-coursework/Dataset/MELBOURNE_HOUSE_PRICES_LESS.csv')

        # Removes samples without a price this includes undisclosed, missing and houses that didn't sell
        # This takes us from 63020 samples down to 48433 samples 
        subset = house_data[house_data["Price"] > 0]
        
        subset = subset.reset_index()
        subset["Address"] = subset["Address"].str.split(" ", n = 1, expand = True)[1]

        # Adjusts the address as to only display the street same so we can compare house prices based on street
        new = subset["Address"].str.split(" ",n = 1, expand = True)
        subset["Address"]=new[1]

        # Removes the data column for most of the categorical data
        subset = subset.drop([ "Address", "Method", "SellerG", "Postcode", "Regionname", "Propertycount", "CouncilArea"], axis=1) 
        
        # Calls date to numerical converison function and replaces dates with new numerical values
        date_values = self.dates_to_numerical(subset["Date"])
        subset["Date"] = np.asarray(date_values)
        
        # Remove data for after a certain date to be used as validation 
        # CODE HERE
        
        split = train_test_split(subset, test_size = 0.15, random_state = 42)
        (x_train, x_test) = split
        
        # Normalises Price seperately so we can easily convert our outputs back later to get meaningfull results
        highest_price = subset["Price"].max()
        y_train = x_train["Price"] / highest_price
        y_test = x_test["Price"] / highest_price
        
        x_train, x_test = self.data_preprocessing(x_train, x_test, subset)
        
        return x_train, y_train, x_test, y_test
        
    # Converts the dates to numerical vales starting from 01/01/2016
    def dates_to_numerical(self, dates):
        dates_list = [dt.datetime.strptime(date, '%d/%m/%Y').date() for date in dates]
        
        start_date = dt.date(2016, 1, 1)
        numerical_dates = []
        for i in enumerate(dates_list):
            a = i[1]-start_date
            numerical_dates.append(a.days)
            
        return numerical_dates
    
    def data_preprocessing(self, x_train, x_test, subset):   
        # Catergorical data normalisation        
        zipBinarizer = LabelBinarizer().fit(subset["Suburb"])
        suburbs_train = zipBinarizer.transform(x_train["Suburb"])
        suburbs_test = zipBinarizer.transform(x_test["Suburb"])
        print(suburbs_train.shape)
        
        zipBinarizer = LabelBinarizer().fit(subset["Type"])
        types_train = zipBinarizer.transform(x_train["Type"])
        types_test = zipBinarizer.transform(x_test["Type"])
        print(types_train.shape)

        # Continuous data normalisation      
        cs = MinMaxScaler()
        continuous_train = cs.fit_transform(x_train[["Rooms", "Date", "Distance"]])
        continuous_test = cs.fit_transform(x_test[["Rooms", "Date", "Distance"]])   
        
        x_train = np.hstack([suburbs_train, types_train, continuous_train])
        x_test = np.hstack([suburbs_test, types_test, continuous_test])
        
        return x_train, x_test 
        
        '''
        x_train_C1 = suburbs_train
        x_train_C2 = types_train
        x_train_Ns = continuous_train
        
        return x_train_C1, x_train_C2, x_train_Ns
        '''

class NeuralNetwork():
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        
        
    def training_loop(self):
        #x_train_categoricalA = self.x_train["Suburb"]
        #x_train_categoricalB = self.x_train["Type"]
        #x_train_numericals = self.x_train["Rooms", "Date", "Distance"]
        
        #model = tf.keras.models.Sequential()
        

        return xNs




create = initialise_data()
xC1,xC2,xNs = create.data_loader() 

runNN = NeuralNetwork(xC1,xC2,xNs)
numericals = runNN.training_loop()
print(numericals)



'''
print(a)
print(b)
print(len(a))
print(len(b))  
'''






