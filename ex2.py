import numpy as np
import logging
import sys
from sklearn.feature_selection import VarianceThreshold


logging.basicConfig(filename='app.log', filemode='w', level=logging.DEBUG, format='%(process)d - %(levelname)s -'
                                                                                  ' %(message)s')

# numerical encoding
sex2int_enc = {'M': 0, 'F': 1, 'I': 2}
# one-hot encoding
sex2one_hot_enc = {'M': np.array([1, 0, 0]), 'F': np.array([0, 1, 0]), 'I': np.array([0, 0, 1])}
LABELS_SIZE = 3
EPOCHS = 30
LR = 0.01


def extract_examples(train_x_path):
    logging.info('{}: start extracting weights from {}'.format(extract_examples.__name__, train_x_path))
    examples = []
    with open(train_x_path) as f:
        for line in f.readlines():
            example_values = line.strip().split(',')
            example_values[0] = str(sex2int_enc[example_values[0]])
            examples.append(np.array(example_values, dtype=float))

    return np.array(examples)


def extract_labels(train_y_path):
    logging.info('{}: start extracting labels from {}'.format(extract_labels.__name__, train_y_path))
    return np.loadtxt(train_y_path, dtype=int)


def min_max_normalize(examples):
    logging.info('{}: start normalizing examples values via Min-Max...'.format(min_max_normalize.__name__))

    max_values = np.max(examples, axis=0)
    min_values = np.min(examples, axis=0)

    rows, columns = examples.shape
    for column in range(columns):
        if max_values[column] - min_values[column] == 0:  # zero division check
            logging.warning('{}: zero division occurred at column = {}...'.format(min_max_normalize.__name__, column))
            continue
        examples[:, column] = (examples[:, column] - min_values[column]) / \
                              (max_values[column] - min_values[column])


def z_score_normalize(examples):
    logging.info('{}: start normalizing examples values via Z-Score...'.format(z_score_normalize.__name__))

    mean_values = np.mean(examples, axis=0)
    std_dev_values = np.std(examples, axis=0)

    rows, columns = examples.shape
    for column in range(columns):
        if std_dev_values[column] == 0:  # zero division check
            logging.warning('{}: zero division occurred at column = {}...'.format(z_score_normalize.__name__, column))
            continue
        examples[:, column] = (examples[:, column] - mean_values[column]) / (std_dev_values[column])


def main(argv):

    train_x_path = argv[0]
    train_y_path = argv[1]
    test_x_path = argv[2]

    logging.info('{}: start extracting data...'.format(main.__name__))
    examples = extract_examples(train_x_path)
    logging.debug('{}: examples dimensions are {}'.format(main.__name__, examples.shape))
    logging.debug(np.array2string(examples))
    labels = extract_labels(train_y_path)
    logging.debug('{}: labels dimensions are {}'.format(main.__name__, labels.shape))
    logging.debug(labels)
    logging.info('{}: done extracting data...'.format(main.__name__))

    logging.info('{}: start normalizing examples...'.format(main.__name__))
    min_max_normalize(examples)
    # z_score_normalize(examples)
    logging.debug(np.array2string(examples))
    logging.info('{}: done normalizing examples...'.format(main.__name__))

    n_examples, n_features = examples.shape
    w = np.zeros((LABELS_SIZE, n_features))

    count = 0
    for epoch in range(EPOCHS):

        for example, gold_label in zip(examples, labels):
            predicted_label = np.argmax(np.dot(w, example))

            if gold_label != predicted_label:
                count += 1
                w[gold_label, :] = w[gold_label, :] + LR * example
                w[predicted_label, :] = w[predicted_label, :] - LR * example

        print("{}".format(count))
        count = 0


if __name__ == '__main__':
    params = sys.argv[1:]

    if not params or len(params) < 3:
        logging.error('{}: not enough arguments passed to {}'.format(main.__name__, 'ass2'))
        exit(1)

    main(params)
