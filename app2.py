import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from PIL import Image,ImageFilter,ImageEnhance
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler



def main() :
    st.title('자동차 구매 예측')
    #사이드바용 메뉴
    menu = ['DataSet','Predictions','About']
    choice = st.sidebar.selectbox("메뉴",menu)

    if choice == 'DataSet' :
        st.write("다음의 데이터 셋을 이용합니다.")
        df = pd.read_csv('Car_Purchasing_Data.csv')
        st.dataframe(df)
    
    elif choice == 'Predictions' :
        #저장된 모델을 불러옵니다.

        car_df = pd.read_csv('Car_Purchasing_Data.csv')
        X= car_df.iloc[:,3:7+1]
        y = car_df['Car Purchase Amount']
        ms = MinMaxScaler()
        X = ms.fit_transform(X)
        y = np.array(y)
        y = y.reshape(-1,1)
        msy = MinMaxScaler()
        y = msy.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.25,random_state=50)
        
        classifier = Sequential()

        classifier.add( Dense(input_dim = 5 , units= 3, activation='relu' )  ) 

        classifier.add(Dense( units= 12 , activation='relu') )

        classifier.add(Dense( units= 8 , activation='relu') )

        classifier.add( Dense(units=1 , activation='linear')) # Linear : 수치 예측일 때는 linear!! 
        classifier.compile(loss='mean_squared_error',optimizer='adam')

        classifier.fit(X_train,y_train, epochs=200)

        


        new_data = ([0,38,90000,2000,500000])
        new_data = np.array(new_data)
        new_data  = new_data.reshape(1,-1)
        new_data_scaled = ms.transform(new_data)
        aa =  classifier.predict(new_data_scaled)
        result = msy.inverse_transform(aa)
        st.write(result)

        # json_file = open("model.json", "r")
        # loaded_model_json = json_file.read()
        # json_file.close()
        # loaded_model = model_from_json(loaded_model_json)
        # loaded_model.load_weights("model.h5")
        # msy=load('std_scaler.bin')

