import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

class RegressionModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.cat_cols = ['Gender', 'EthnicGroup', 'ParentEduc', 'LunchType', 'TestPrep',
                         'ParentMaritalStatus', 'PracticeSport', 'IsFirstChild', 'TransportMeans', 'WklyStudyHours']
        self.encoder = OrdinalEncoder()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model = RandomForestRegressor(n_estimators=100)
        self.y_pred = None
        self.rmse = None
        self.df_pred_actual = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        self.df.drop(columns=['Unnamed: 0'], inplace=True)
        print(self.df.head())

    def preprocess_data(self):
        self.df.dropna(inplace=True)
        self.df[self.cat_cols] = self.encoder.fit_transform(self.df[self.cat_cols])
        print(self.df.head())

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df.drop(columns=['MathScore', 'ReadingScore', 'WritingScore']),
            self.df[['MathScore', 'ReadingScore', 'WritingScore']],
            test_size=0.2,
            random_state=40
        )

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)

    def calculate_rmse(self):
        self.rmse = mean_squared_error(self.y_test, self.y_pred, squared=False)
        print('RMSE:', self.rmse)

    def analyze_results(self):
        self.df_pred_actual = pd.DataFrame({
            'Actual Math Score': self.y_test['MathScore'].values,
            'Predicted Math Score': self.y_pred[:, 0],
            'Actual Reading Score': self.y_test['ReadingScore'].values,
            'Predicted Reading Score': self.y_pred[:, 1],
            'Actual Writing Score': self.y_test['WritingScore'].values,
            'Predicted Writing Score': self.y_pred[:, 2]
        })
        print(self.df_pred_actual.head(10))

    def visualize_predictions(self):
        sns.set_style("darkgrid")
        plt.figure(figsize=(12, 4))
        
        # Math Scores
        plt.subplot(1, 3, 1)
        plt.scatter(self.y_test['MathScore'], self.y_pred[:, 0])
        plt.plot([0, 100], [0, 100], "--", color="red")
        plt.xlim([0, 100])
        plt.ylim([0, 100])
        plt.xlabel("Actual Math Scores")
        plt.ylabel("Predicted Math Scores")
        plt.title("Predicted vs Actual Math Scores (RMSE={:.2f})".format(self.rmse))
        
        # Reading Scores
        plt.subplot(1, 3, 2)
        plt.scatter(self.y_test['ReadingScore'], self.y_pred[:, 1])
        plt.plot([0, 100], [0, 100], "--", color="red")
        plt.xlim([0, 100])
        plt.ylim([0, 100])
        plt.xlabel("Actual Reading Scores")
        plt.ylabel("Predicted Reading Scores")
        plt.title("Predicted vs Actual Reading Scores (RMSE={:.2f})".format(self.rmse))
        
        # Writing Scores
        plt.subplot(1, 3, 3)
        plt.scatter(self.y_test['WritingScore'], self.y_pred[:, 2])
        plt.plot([0, 100], [0, 100], "--", color="red")
        plt.xlim([0, 100])
        plt.ylim([0, 100])
        plt.xlabel("Actual Writing Scores")
        plt.ylabel("Predicted Writing Scores")
        plt.title("Predicted vs Actual Writing Scores (RMSE={:.2f})".format(self.rmse))
        
        plt.tight_layout()
        plt.show()
    
    def visualize_feature_importance(self):
        feat_importances = pd.Series(self.model.feature_importances_, index=self.X_train.columns)
        feat_importances.nlargest(10).plot(kind='barh')
        plt.xlabel("Feature Importance")
        plt.ylabel("Features")
        plt.title("Top 10 Important Features")
        plt.show()


model = RegressionModel('Expanded_data_with_more_features.csv')
model.load_data()
model.preprocess_data()
model.split_data()
model.train_model()
model.predict()
model.calculate_rmse()
model.analyze_results()
model.visualize_predictions()
model.visualize_feature_importance()
