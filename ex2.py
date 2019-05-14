import numpy as np
import sys

sex2int_enc = {'M': 0, 'F': 1, 'I': 2}                                                  # numeric encoding
sex2one_hot_enc = {'M': ['1', '0', '0'], 'F': ['0', '1', '0'], 'I': ['0', '0', '1']}    # one-hot encoding

# consts
LABELS_SIZE = 3
HINGE_LOSS_MARGIN = 1


class Model:
    """
    Abstract Model class
    """
    def __init__(self, train_x, train_y, test_x):
        self.train_x, self.train_y, self.test_x = train_x, train_y, test_x
        self.n_examples, self.n_features = self.train_x.shape
        self.W = np.zeros((LABELS_SIZE, self.n_features))
        # self.W = np.random.uniform(-.5, .5, [LABELS_SIZE, self.n_features])   # sample uniformly in range [-.5, .5]

    def train(self, epochs):
        """
        training the model for fixed number of epochs
        :param epochs: number of iterations till stop training
        :return: None
        """
        for epoch in range(epochs):
            # self.shuffle()
            for x, y in zip(self.train_x, self.train_y):
                scores = self.score(x)
                y_hat = self.predict(scores)
                if y != y_hat:
                    hinge_loss = self.hinge_loss(scores, y, y_hat)
                    self.update(x, y, y_hat, hinge_loss)

    @staticmethod
    def hinge_loss(scores, y, y_hat):
        """
        calculating hinge loss on error only when model predicted not correct
        :param scores: scores of each one of the labels
        :param y: gold label
        :param y_hat: gold label
        :return: hinge loss
        """
        return max(0, HINGE_LOSS_MARGIN - scores[y] + scores[y_hat])

    def shuffle(self):
        """
        shuffle the training dataset at each epoch
        :return: None
        """
        map_index_position = list(zip(self.train_x, self.train_y))
        np.random.shuffle(map_index_position)
        self.train_x, self.train_y = zip(*map_index_position)

    def score(self, x):
        """
        scoring each label of the model on example
        :param x: example
        :return: numpy array of scores
        """
        return np.dot(self.W, x)

    @staticmethod
    def predict(scores):
        """
        predicting the best label from the scores (by taking the index of the highest score)
        :param scores: scores of each label
        :return: predicted label
        """
        return np.argmax(scores)

    def update(self, x, y, y_hat, loss):
        """
        abstract function which all his sons implementing
        :param x: example
        :param y: gold label
        :param y_hat: prediction label
        :param loss: loss on error
        :return: None
        """
        pass

    def test(self):
        """
        testing the model on test dataset
        :return: list of predictions of the model on test dataset
        """
        predictions = []
        for x in self.test_x:
            scores = self.score(x)
            y_hat = self.predict(scores)
            predictions.append(y_hat)
        return predictions


class Perceptron(Model):
    """ Perceptron Model extending from Model """
    def __init__(self, train_x, train_y, dev_x, eta):
        super().__init__(train_x, train_y, dev_x)
        self.eta = eta

    def update(self, x, y, y_hat, loss):
        """
        svm updating weights
        :param x: example
        :param y: gold label
        :param y_hat: predicted label
        :param loss: loss on error
        :return: None
        """
        self.W[y, :] = self.W[y, :] + self.eta * x
        self.W[y_hat, :] = self.W[y_hat, :] - self.eta * x


class SVM(Model):
    """ SVM Model extending from Model """
    def __init__(self, train_x, train_y, dev_x, eta, lamb):
        super().__init__(train_x, train_y, dev_x)
        self.eta = eta
        self.lamb = lamb

    def update(self, x, y, y_hat, loss):
        """
        svm updating weights
        :param x: example
        :param y: gold label
        :param y_hat: predicted label
        :param loss: loss on error
        :return: None
        """
        self.W[y, :] = (1 - self.eta * self.lamb) * self.W[y, :] + self.eta * x
        self.W[y_hat, :] = (1 - self.eta * self.lamb) * self.W[y_hat, :] - self.eta * x
        self.W[LABELS_SIZE - y - y_hat, :] = \
            (1 - self.eta * self.lamb) * self.W[LABELS_SIZE - y - y_hat, :]


class PA(Model):
    """ PA Model extending from Model """
    def __init__(self, train_x, train_y, dev_x):
        super().__init__(train_x, train_y, dev_x)

    def update(self, x, y, y_hat, loss):
        """
        passive aggressive updating weights
        :param x: example
        :param y: gold label
        :param y_hat: predicted label
        :param loss: loss on error
        :return: None
        """
        tau = loss / (2 * np.square(np.linalg.norm(x)))
        self.W[y, :] = self.W[y, :] + tau * x
        self.W[y_hat, :] = self.W[y_hat, :] - tau * x


def extract_features(path_to_features):
    """
    extracting features from features file
    :param path_to_features: path to features file
    :return: numpy array of features (float type)
    """
    features = []
    with open(path_to_features) as f:
        for line in f.readlines():
            feature_values = line.strip().split(',')
            feature_values[0] = str(sex2int_enc[feature_values[0]])  # categorical  to numeric encoding
            # feature_values = sex2one_hot_enc[feature_values[0]] + feature_values[1:]  # to one hot encoding
            features.append(np.array(feature_values, dtype=np.float))

    return np.array(features)


def extract_labels(path_to_labels):
    """
    extracting labels from labels file
    :param path_to_labels: path to labels file
    :return: numpy array of labels (int32 type)
    """
    return np.loadtxt(path_to_labels, dtype=np.int32)


def extract_min_max_values(features, axis=0):
    """
    extracting min & max values of each feature from features
    :param features: features we extracting from (train features)
    :param axis: axis from which we extract the values (default=0=columns)
    :return: min & max values
    """
    return np.max(features, axis=axis), np.min(features, axis=axis)


def extract_mean_dev_values(features, axis=0):
    """
    extracting mean & standard values of each feature from features
    :param features: features we extracting from (train features)
    :param axis: axis from which we extract the values (default=0=columns)
    :return: mean & dev values
    """
    return np.mean(features, axis=axis), np.std(features, axis=axis)


def min_max_normalizer(features, max_values, min_values):
    """
    normalizing features by min-max normalization
    :param features: features we want to normalize
    :param max_values: min values of each features extracted from train dataset
    :param min_values: amx values of each features extracted from train dataset
    :return: None
    """
    n_examples, n_features = features.shape
    for column in range(n_features):
        if max_values[column] - min_values[column] == 0:            # zero division check
            continue
        features[:, column] = (features[:, column] - min_values[column]) / \
                              (max_values[column] - min_values[column])


def z_score_normalizer(features, mean_values, std_dev_values):
    """
    normalizing features by z-score normalization
    :param features: features we want to normalize
    :param mean_values: mean values of each features extracted from train dataset
    :param std_dev_values: standard deviation values of each feature extracted from train dataset
    :return: None
    """
    n_examples, n_features = features.shape
    for column in range(n_features):
        if std_dev_values[column] == 0:                             # zero division check
            continue
        features[:, column] = (features[:, column] - mean_values[column]) / \
                              (std_dev_values[column])


def train_models(models):
    """
    training models on train dataset
    :param models: models we want to train
    :return: None
    """
    for model_name in ['Perceptron', 'SVM', 'PA']:
        model_info = models[model_name]
        model = model_info['model']
        model.train(model_info['epochs'])


def test_models(models):
    """
    testing models on test dataset
    :param models: models we want to test on
    :return: models' predictions on test dataset
    """
    models_predictions = {}
    for model_name in ['Perceptron', 'SVM', 'PA']:
        model_info = models[model_name]
        model = model_info['model']
        models_predictions[model_name] = model.test()
    return models_predictions


def print_predictions(models_predictions):
    """
    printing model's predictions on test dataset
    :param models_predictions:  map of model to his predictions on test dataset
    :return: None
    """
    perceptron_predictions = models_predictions['Perceptron']
    svm_predictions = models_predictions['Perceptron']
    pa_predictions = models_predictions['Perceptron']

    test_length = len(perceptron_predictions)
    for i in range(test_length):
        print('perceptron: {}, svm: {}, pa: {}'.format(perceptron_predictions[i],
                                                       svm_predictions[i], pa_predictions[i]))


def main(argv):
    """
    main function which extracting/normalizing data, builds linear models, training them on train dataset,
    and testing the models on test dataset, finally printing the outputs
    :param argv: applications' arguments from the user
    :return: None
    """
    # path to train and test files
    train_x_path = argv[0]
    train_y_path = argv[1]
    test_x_path = argv[2]

    # extracting train data/labels
    train_features = extract_features(train_x_path)
    train_labels = extract_labels(train_y_path)

    # extracting test data
    text_features = extract_features(test_x_path)

    # preprocessing train/test data
    max_values, min_values = extract_min_max_values(train_features)
    min_max_normalizer(train_features, max_values, min_values)
    min_max_normalizer(text_features, max_values, min_values)
    # mean_values, dev_values = extract_mean_dev_values(train_features)
    # z_score_normalizer(train_features, mean_values, dev_values)
    # z_score_normalizer(text_features, mean_values, dev_values)

    # creating linear models
    models = {'Perceptron': {'model': Perceptron(train_features, train_labels, text_features, eta=0.001), 'epochs': 15},
              'SVM': {'model': SVM(train_features, train_labels, text_features, eta=0.001, lamb=0.001), 'epochs': 15},
              'PA': {'model': PA(train_features, train_labels, text_features), 'epochs': 5}}

    train_models(models)                      # train models
    models_predictions = test_models(models)  # test models
    print_predictions(models_predictions)     # print predictions


if __name__ == '__main__':
    if len(sys.argv[1:]) < 3:   # checking if enough arguments provided by user
        exit(1)

    main(sys.argv[1:])          # run main
    exit(0)
