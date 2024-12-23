import joblib


class EnsembleModel:
    def __init__(self, model_name, weight_path):
        """
        Khởi tạo mô hình ensemble với tên và đường dẫn đến file trọng số.
        :param model_name: Tên của mô hình (str).
        :param weight_path: Đường dẫn đến file chứa trọng số (str).
        """
        self.model_name = model_name
        self.model = joblib.load(weight_path)

    def get_name(self):
        """Trả về tên của mô hình."""
        return self.model_name

    def predict(self, X):
        """
        Dự đoán kết quả với dữ liệu đầu vào.
        :param X: Dữ liệu đầu vào (list hoặc numpy array).
        :return: Kết quả dự đoán.
        """
        return self.model.predict(X)


if __name__ == "__main__":
    # Tạo đối tượng EnsembleModel
    model = EnsembleModel(
        "random_forest_SongChay3",
        "/home/jackson-devworks/Desktop/raining_prediction/weights/SongChay3_random_forest.pkl",
    )

    # Dữ liệu đầu vào
    inputs = [[0, 0.113, 0.222, 2.314, 2.812]]

    # Dự đoán
    output = model.predict(inputs)

    # In kết quả
    print(f"Tên mô hình: {model.get_name()}")
    print(f"Dự đoán: {output}")
