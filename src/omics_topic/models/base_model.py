class BaseModel:
    """
    Base class for all models in the omics_topic package.

    This class provides a common interface and shared functionality for all models.
    It can be extended by specific model implementations.
    """

    def __init__(self):
        pass

    def fit(self, data):
        """
        Fit the model to the provided data.

        Parameters
        ----------
            data: The input data to fit the model.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def predict(self, data):
        """
        Predict using the fitted model on the provided data.

        Parameters
        ----------
            data: The input data for prediction.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
