import pandas as pd 
# Makes sure we see all columns
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids

class Cervical_DataLoader():
    def __init__(self):
        self.data = None

    def load_dataset(self, path="data/cervical.csv"):
        self.data = pd.read_csv(path)

    def preprocess_data(self):
        # 因為 STDs: Time since first diagnosis 和 STDs: Time since last diagnosis 缺少太多資料，因此刪除此兩列特徵
        self.data.drop(['STDs..Time.since.first.diagnosis','STDs..Time.since.last.diagnosis'], inplace=True, axis=1)
        
        # 將 lable mapping 到 0 和 1
        self.data['Biopsy'] = self.data['Biopsy'].map({'Healthy':0, 'Cancer':1})
        
        # 分別列出數值變數名稱和類別變數名稱
        num_cols = ['Age', 'Number.of.sexual.partners', 'First.sexual.intercourse', 'Num.of.pregnancies', 'Smokes..years.',
                    'Hormonal.Contraceptives..years.', 'IUD..years.','STDs..number.', 'STDs..Number.of.diagnosis']
        cat_cols = ['Smokes', 'Hormonal.Contraceptives', 'IUD', 'STDs']
        
        # 對類別變數進行 one hot encoding
        self.data = pd.get_dummies(data=self.data, columns=cat_cols)

    def get_data_split(self):
        y = self.data.pop('Biopsy')
        X = self.data.copy()
        return train_test_split(X, y, test_size=0.20, random_state=2022)
    
    def oversample(self, X_train, y_train):
        oversample = SMOTE(random_state=2022)
        # Convert to numpy and oversample
        x_np = X_train.to_numpy()
        y_np = y_train.to_numpy()
        x_np, y_np = oversample.fit_resample(x_np, y_np)
        # Convert back to pandas
        x_over = pd.DataFrame(x_np, columns=X_train.columns)
        y_over = pd.Series(y_np, name=y_train.name)
        return x_over, y_over
    
    def undersample(self, X_train, y_train):
        undersample = ClusterCentroids(random_state=2022)
        # Convert to numpy and oversample
        x_np = X_train.to_numpy()
        y_np = y_train.to_numpy()
        x_np, y_np = undersample.fit_resample(x_np, y_np)
        # Convert back to pandas
        x_under = pd.DataFrame(x_np, columns=X_train.columns)
        y_under = pd.Series(y_np, name=y_train.name)
        return x_under, y_under
    
class Bike_DataLoader():
    def __init__(self):
        self.data = None

    def load_dataset(self, path="data/bike.csv"):
        self.data = pd.read_csv(path)

    def preprocess_data(self):
        # 分別列出數值變數名稱和類別變數名稱
        num_cols = ['temp', 'hum', 'windspeed', 'days_since_2011']
        cat_cols = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
        
        # 對類別變數進行 one hot encoding
        self.data = pd.get_dummies(data=self.data, columns=cat_cols)

    def get_data_split(self):
        y = self.data.pop('cnt')
        X = self.data.copy()
        return train_test_split(X, y, test_size=0.20, random_state=2022)
    
    def oversample(self, X_train, y_train):
        oversample = SMOTE(random_state=2022)
        # Convert to numpy and oversample
        x_np = X_train.to_numpy()
        y_np = y_train.to_numpy()
        x_np, y_np = oversample.fit_resample(x_np, y_np)
        # Convert back to pandas
        x_over = pd.DataFrame(x_np, columns=X_train.columns)
        y_over = pd.Series(y_np, name=y_train.name)
        return x_over, y_over
    
    def undersample(self, X_train, y_train):
        undersample = ClusterCentroids(random_state=2022)
        # Convert to numpy and oversample
        x_np = X_train.to_numpy()
        y_np = y_train.to_numpy()
        x_np, y_np = undersample.fit_resample(x_np, y_np)
        # Convert back to pandas
        x_under = pd.DataFrame(x_np, columns=X_train.columns)
        y_under = pd.Series(y_np, name=y_train.name)
        return x_under, y_under