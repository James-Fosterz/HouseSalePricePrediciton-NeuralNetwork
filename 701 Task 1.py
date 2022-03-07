import pandas as pd
import numpy as np
import datetime as dt
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#from time import time

# Neural Network model
input_layer_size = 376
hidden_layers_sizes = [128, 64, 32]
output_layer_size = 1

model = nn.Sequential(nn.Linear(input_layer_size, hidden_layers_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_layers_sizes[0], hidden_layers_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_layers_sizes[1], hidden_layers_sizes[2]),
                      nn.ReLU(),
                      nn.Linear(hidden_layers_sizes[2], output_layer_size),
                      nn.ReLU()
                      )


class initialise_data():
    def data_loader(self):      
        house_data = pd.read_csv('Dataset/MELBOURNE_HOUSE_PRICES_LESS.csv',usecols = [0,2,3,4,7,11])
    
        # Removes samples without a price this includes undisclosed, missing and houses that didn't sell
        # This takes us from 63020 samples down to 48433 samples 
        subset = house_data[house_data["Price"] > 0]
        
        subset = subset.reset_index()
                
        # Calls date to numerical converison function and replaces dates with new numerical values
        date_values = self.dates_to_numerical(subset["Date"])
        subset["Date"] = np.asarray(date_values)
        
        split = train_test_split(subset, test_size = 0.15, random_state = 42)
        (x_train, x_test) = split
        
        # Normalises Price seperately so we can easily convert our outputs back later to get meaningfull results
        highest_price = subset["Price"].max()
        y_train = x_train["Price"] / highest_price
        y_train = y_train.values.reshape(-1,1)
        y_test = x_test["Price"] / highest_price
        y_test = y_test.values.reshape(-1,1)
        
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
        #print(suburbs_train.shape)
        
        zipBinarizer = LabelBinarizer().fit(subset["Type"])
        types_train = zipBinarizer.transform(x_train["Type"])
        types_test = zipBinarizer.transform(x_test["Type"])
        #print(types_train.shape)

        # Continuous data normalisation      
        cs = MinMaxScaler()
        continuous_train = cs.fit_transform(x_train[["Rooms", "Date", "Distance"]])
        continuous_test = cs.fit_transform(x_test[["Rooms", "Date", "Distance"]])   
        
        x_train = np.hstack([suburbs_train, types_train, continuous_train])
        x_test = np.hstack([suburbs_test, types_test, continuous_test])
        
        return x_train, x_test 


class NeuralNetwork():
    def __init__(self, x_train, y_train, x_test, y_test, epochs):
        self.x_train = torch.from_numpy(np.asarray(x_train))      
        self.y_train = torch.from_numpy(np.asarray(y_train))
        self.x_test = torch.from_numpy(np.asarray(x_test))
        self.y_test = torch.from_numpy(np.asarray(y_test))
        self.epochs = epochs
        
        
        

        
    def training_loop(self):
        print(self.x_train.dtype)
        
        loss = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.2)
        
        model.eval()
        pred_val = model(self.x_train.float())
        before_train = loss(pred_val.flatten(), self.y_train.flatten())
        print("Test loss before training", before_train.item())
        
        model.train()
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch}")
            optimizer.zero_grad()
                    
            #Forward pass
            pred_vals = model(self.x_train.float())
                 
            #Calculating the Loss
            output = loss(pred_vals.flatten(), self.y_train.flatten().float())
                
            print('Epoch {}: train loss: {}'.format(epoch, output.item()))
            print(f"PRINTING Predicted VALUE {pred_vals.data[0].flatten()}")
            print(f"PRINTING Y LABEL         {self.y_train[0]}")
       
            #Backward pass
            output.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            predicted_value = model(self.x_test.float())
        post_train = loss(predicted_value.float(), self.y_test.float())
        print("The loss after Training", post_train.item())
        
        return 


create = initialise_data()
x_train, y_train, x_test, y_test = create.data_loader()


epochs = 50

runNN = NeuralNetwork(x_train,y_train, x_test, y_test, epochs)
numericals = runNN.training_loop()

