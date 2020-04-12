from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Flatten, BatchNormalization, Dropout, Conv2D, MaxPooling2D
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.optimizers import Adam
from PIL import Image
from skimage.exposure import equalize_adapthist as eq_hist
import pickle


#df = pd.read_csv('oasis_longitudinal.csv')
#df["SES"].fillna(df["SES"].median(), inplace=True)
#df["MMSE"].fillna(df["MMSE"].Smean(), inplace=True)
#df=df.dropna(axis=0,how="any")
#
#columns = ['Subject ID', 'MRI ID', 'Group', 'Visit', 'MR Delay', 'Hand', 'CDR']
##df = df.drop(columns, axis = 1)

IMAGE_SIZE = 176
 
class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("Alzheimer MRI INPUT IMAGE")
        self.minsize(640, 400)
        self.model = load_model('model.hdf5')
 
        self.labelFrame = ttk.LabelFrame(self, text = "Open File")
        self.labelFrame.grid(column = 0, row = 1, padx = 20, pady = 20)
 
        self.button()
        self.label = Label(self, text = "You are detected with dementia, Please Browse PET scan image for Further Analysis").grid(column = 0, row = 5)
 
 
 
    def button(self):
        self.button = ttk.Button(self.labelFrame, text = "Browse A File",command = self.fileDialog)
        self.button.grid(column = 1, row = 1)
 
 
    def fileDialog(self):
 
        self.filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype =
        (("jpeg files","*.jpg"),("all files","*.*")) )
        self.label = ttk.Label(self.labelFrame, text = "")
        self.label.grid(column = 1, row = 2)
        self.label.configure(text = self.filename)
        #print(self.filename*2)
        im = Image.open(self.filename).convert('L')
        im = im.resize((176,176))
        self.im = np.array(eq_hist(np.array(im), clip_limit=0.03))
        self.im = self.im.reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 1))
        self.lmodel()
        
        
    def lmodel(self):
        #self.model = self.create_cnn(filters=[32, 64, 128, 256], kernels=[3, 3, 5], denses=[64, 128], dropout=0.5, reg=.0005)
        #self.model.predict()
        #self.model = load_model('model.hdf5')
        #print(self.model.predict(self.im))
        prediction = self.model.predict(self.im)
        print(prediction)
        k = np.where(np.amax(prediction))
        #k = prediction.index(j)
        if(k == 0):
            #print('NonDemented')
            label = Label(self, text = 'NonDemented').grid(column = 0, row = 7)
        elif(k == 1):
            #print('VeryMildDemented')
            label = Label(self, text = 'VeryMildDemented').grid(column = 0, row = 7)
        elif(k == 2):
            #print('MildDemented')
            label = Label(self, text = 'MildDemented').grid(column = 0, row = 7)
        else:
            #print('ModerateDemented')
            label = Label(self, text = 'ModerateDemented').grid(column = 0, row = 7)
                           

 
 

class Master(Tk):
    def __init__(self):
        super(Master,self).__init__()
        self.title("Alzeimer's detector")
        self.gender_label = Label(self, text = "Gender")
        self.gender_combo = ttk.Combobox(self, values = ['male','female'])


        self.age_label = Label(self, text = "age")
        self.age_tb = Entry(self)


        self.educ_label = Label(self, text = "EDUC")
        self.educ_tb = Entry(self)


        self.ses_label = Label(self, text = "SES")
        self.ses_tb = Entry(self)

        self.mmse_label = Label(self, text = 'MMSE')
        self.mmse_tb = Entry(self)


        self.etiv_label = Label(self, text = 'eTIV')
        self.etiv_tb = Entry(self)


        self.nwbv_label = Label(self, text = 'nWBV')
        self.nwbv_tb = Entry(self)

        self.asf_label = Label(self, text = 'ASF')
        self.asf_tb = Entry(self)
        self.b = Button(self, text="Check", command=self.get_values)
        self.grids()
        
    
    

        
           

            
 
    def get_values(self):
        if(self.gender_combo.get() == 'male'):
            self.gender = 1
        else:
            self.gender = 0
        self.age = int(self.age_tb.get())
        self.educ = int(self.educ_tb.get())
        self.ses = int(self.ses_tb.get())
        self.mmse = int(self.mmse_tb.get())
        self.etiv = int(self.etiv_tb.get())
        self.nwbv = float(self.nwbv_tb.get())
        self.asf = float(self.asf_tb.get())
        '''d = {
            'M/F':[self.gender],
            'Age':[self.age],  
            'EDUC':[self.educ],  
            'SES':[self.ses],  
            'MMSE':[self.mmse],  
            'eTIV':[self.etiv],
            'nWBV':[self.nwbv],
            'ASF':[self.asf]}'''
        d=[(self.gender,self.age,self.educ,self.ses,self.mmse,self.etiv,self.nwbv,self.asf)]
        feature_col_names = ["M/F", "Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF"]

        
        temp = pd.DataFrame(d, columns = feature_col_names)
        print(temp)
        #df1=pd.concat([df,temp])
        #temp['M/F'] = temp['M/F'].replace(['F','M'], [0,1])
        #print(self.gender, self.age, self.asf)
        #print(df1.tail(5))
        #print(df1.tail(5))
        #X=df1[feature_col_names].values
        #print(X)
        with open('rfm.model','rb') as f:
            model2 = pickle.load(f)
            j = model2.predict(temp.values)
            if(j == [1]):
                self.create_window()
                
            #print(j)
        

    
        

    '''def create_cnn(filters=[32], kernels=[3, 3, 5], dropout=0.5, denses=[128], reg=.0001):

        model = Sequential()
    
        for i, fil in enumerate(filters):
            if i == 0:
                model.add(Conv2D(fil, kernels[0], activation='elu', padding='same', kernel_regularizer=l2(reg), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
            else:
                model.add(Conv2D(fil, kernels[0], activation='elu', padding='same', kernel_regularizer=l2(reg)))
        
            for ker in kernels[1:]:
                model.add(BatchNormalization())
                model.add(Conv2D(fil, ker, activation='elu', padding='same', kernel_regularizer=l2(reg)))
            
            model.add(MaxPooling2D())
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        model.add(Flatten())
    
        for den in denses:
            model.add(Dense(den, activation='elu', kernel_regularizer=l2(reg)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))
    
        model.add(Dense(len(CLASSES), activation='softmax'))

        model.load_weights('/model.hdf5')
        
        model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(.001),
            metrics=['acc']
        )
        

        return model'''

    
    def create_window(self):    
        root = Root()
        #self.lmodel()

    def grids(self):
        self.gender_label.grid(column = 0, row = 1)
        self.gender_combo.grid(column = 1, row = 1)
        self.age_label.grid(column = 0, row = 2)
        self.age_tb.grid(column = 1, row = 2)
        self.educ_label.grid(column = 0, row = 3)
        self.educ_tb.grid(column = 1, row = 3)
        self.ses_label.grid(column = 0, row = 4)
        self.ses_tb.grid(column = 1, row = 4)
        self.mmse_label.grid(column = 0, row = 5)
        self.mmse_tb.grid(column = 1, row = 5)
        self.etiv_label.grid(column = 0, row = 6)
        self.etiv_tb.grid(column = 1, row = 6)
        self.nwbv_label.grid(column = 0, row = 7)
        self.nwbv_tb.grid(column = 1, row = 7)
        self.asf_label.grid(column = 0, row = 8)
        self.asf_tb.grid(column = 1, row = 8)
        self.b.grid(column = 1, row = 9)


########master window#########
    
master = Master()
#master.minsize(640, 400)




#######getting values########







##########grids###########


#master.grid_columnconfigure(4, minsize=100)

master.mainloop()




