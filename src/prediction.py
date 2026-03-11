import numpy as np
import pickle


class InusrancePrediction:
    def __init__(self):
        with open('artifacts/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        with open ('artifacts/model.pkl', 'rb') as f:
            self.model = pickle.load(f)

    def prediction(self, Age, Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs):
        inp = np.array([[Age, Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs]])
        scaled_input = self.scaler.transform(inp)
        result = self.model.predict(scaled_input)
        return result[0]