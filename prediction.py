import numpy as np

class SimpleLinearRegression:  # LinearRegression
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.hstack([np.ones((X.shape[0], 1)), X])  # add bias term for intercept
        beta = np.linalg.lstsq(X, y, rcond=None)[0]  # solve for coefficients
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]

    def predict(self, X):
        return self.intercept_ + np.dot(X, self.coef_)

def train_prediction_model(df, property_type):  
    # summing columns for tax collection and demand
    tax_sums = {
        "collection": df.select([
            f'Tax_Collection_Cr_2013_14_{property_type}',
            f'Tax_Collection_Cr_2014_15_{property_type}',
            f'Tax_Collection_Cr_2015_16_{property_type}',
            f'Tax_Collection_Cr_2016_17_{property_type}',
            f'Tax_Collection_Cr_2017_18_{property_type}'
        ]).sum().to_numpy()[0],  

        "demand": df.select([
            f'Tax_Demand_Cr_2013_14_{property_type}',
            f'Tax_Demand_Cr_2014_15_{property_type}',
            f'Tax_Demand_Cr_2015_16_{property_type}',
            f'Tax_Demand_Cr_2016_17_{property_type}',
            f'Tax_Demand_Cr_2017_18_{property_type}'
        ]).sum().to_numpy()[0]
    }

    X = np.array([2014, 2015, 2016, 2017, 2018]).reshape(-1, 1)
    models = {}

    for key in ["collection", "demand"]:
        y = tax_sums[key]
        model = SimpleLinearRegression()  
        model.fit(X, y)
        models[key] = model

    def predict_tax(year):  
        if year is not None and year >= 2019:  
            return {
                "predicted_collection": round(models["collection"].predict([[year]])[0], 2),
                "predicted_demand": round(models["demand"].predict([[year]])[0], 2)
            }
        return None

    return predict_tax
