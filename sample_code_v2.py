import joblib


class EnsembleModel:
    def __init__(self, model_name, weight_path):
        """
        Initialize the ensemble model with its name and the path to the weight file.

        :param model_name: Name of the model (str).
        :param weight_path: Path to the file containing the model weights (str).
        """
        self.model_name = model_name  # Store the model name for identification.
        self.model = joblib.load(
            weight_path
        )  # Load the pre-trained model from the specified path.

    def get_name(self):
        """
        Retrieve the name of the model.

        :return: The name of the model (str).
        """
        return self.model_name

    def predict(self, X):
        """
        Predict the output based on the input data using the loaded model.

        :param X: Input data (list or numpy array).
        :return: Predictions made by the model.
        """
        return self.model.predict(X)  # Use the model to make predictions.


if __name__ == "__main__":
    # Create an instance of EnsembleModel.
    model = EnsembleModel(
        "random_forest_SongChay3",  # Name of the model.
        "/home/jackson-devworks/Desktop/raining_prediction/weights/SongChay3_random_forest.pkl",  # Path to the model's weight file.
    )

    # Define the input data for prediction.
    """
        Input details:
            - Time difference = Datetime - Section. Convert int to float
            - List of outputs from raw model. datatype: float

        Specific:
         - Ban Nhung: ["Time difference", "COMS","GFS","MITSUISHI2011_D02","LING3_D02","LINKF_D02","LINBMJ_D02","ETAKF_D02","ETAG3_D02","ETABMJ_D02"]
         - Ban Ve: ["Time difference", "COMS","GFS","MITSUISHI2011_D03","LINKF_D03","LINBMJ_D03","ETAKF_D03","ETAG3_D03","ETABMJ_D03"]
         - Hua Na: ["Time difference", "COMS","GFS","MITSUISHI2011_D03","LINKF_D03","LINBMJ_D03","ETAKF_D03","ETAG3_D03","ETABMJ_D03"]
         - Khanh Khe: ["Time difference", "COMS","GFS","MITSUISHI2011_D02","LING3_D02","LINKF_D02","LINBMJ_D02","ETAKF_D02","ETAG3_D02","ETABMJ_D02"]
         - Muong Hum: ["Time difference", "WRF84H","COMS","GFS","MITSUISHI2011_D01","MITSUISHI2021"]
         - Song Chay 3: ["Time difference", "WRF84H","COMS","GFS","MITSUISHI2011_D01","MITSUISHI2021"]
         - Song Chung: ["Time difference", "WRF84H","COMS","GFS","MITSUISHI2011_D01","MITSUISHI2021"]
         - Thac Xang: ["Time difference", "COMS","GFS","MITSUISHI2011_D02","LING3_D02","LINKF_D02","LINBMJ_D02","ETAKF_D02","ETAG3_D02","ETABMJ_D02"]
    """
    inputs = [[0, 0.113, 0.222, 2.314, 2.812]]  # Example feature values.

    # Make predictions using the model.
    output = model.predict(inputs)

    # Print the model's name and its predictions.
    print(f"Tên mô hình: {model.get_name()}")  # Display the model's name.
    print(f"Dự đoán: {output}")  # Display the predictions.
