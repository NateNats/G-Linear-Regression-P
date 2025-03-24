import numpy as np

class Linear_Regression():
    def __init__(self, alpha=0.00001, n_iter=1000):
        self.alpha = alpha # jangan terlalu besar, nanti nilai w atau b bakal infinite
        self.n_iter = n_iter
        self.params = {}

    def param_init(self, X_train):
        """
        inisialisasi parameter untuk linear regression
        input:
        - X (untuk training data)
        """

        # handle 1d or more columns in X_train
        X_train = X_train.shape[-1, 1] if X_train.ndim == 1 else X_train

        _, n_features = X_train.shape
        self.params['W'] = np.zeros(n_features)
        self.params['b'] = 0

        return self

    
    def gradient_descent(self, X_train, y_train):
        """
        Fungsi gradient descent adalah 
        untuk meminimalkan cost function

        input:
        - X (training data)
        - y (label / target)
        - params
        - alpha

        output:
        - w = slope (menunjukkan seberapa besar perubahan pada y setiap perubahan satu unit pada x)
        - b = intercept (titik potong garis linier)
        - m = data
        """

        # handle 1d or more columns in X_train
        X_train = X_train.shape[-1, 1] if X_train.ndim == 1 else X_train
        
        W = self.params['W']
        b = self.params['b']
        m = X_train.shape[0]

        for _ in range(self.n_iter):
            # mencoba memprediksi dengan bobot random
            y_pred = np.dot(X_train, W) + b

            # partial derivative of coefficients
            db = (2/m) * np.sum(y_pred - y_train)
            dw = (2/m) * np.dot(X_train.T, (y_pred - y_train))


            # update coefficients
            W -= self.alpha * dw
            b -= self.alpha * db

        self.params['W'] = W
        self.params['b'] = b


        return self


    def train(self, X_train, y_train):
        """
        melakukan training model linear regression menggunakan gradient descent
        input:
        - X_train
        - y_train
        - alpha (learning rate)
        - n_iter (jumlah iterasi)
        """
        self.param_init(X_train)
        self.gradient_descent(X_train, y_train)
        return self


    def predict(self, X_test):
        """
        memprediksi data yang diinput
        input:
        - X (data yang belum pernah di train sebelumnya)

        output:
        - y_preds (X yang sudah diprediksi oleh model)
        """
        y_preds = np.dot(X_test, self.params['W']) + self.params['b']
        return y_preds