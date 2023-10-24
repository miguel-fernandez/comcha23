def setup_iris_data():
    from sklearn.datasets import load_iris

    iris = load_iris()

    return iris.data, iris.target


def setup_digits_data():
    from sklearn.datasets import load_digits

    digits = load_digits()

    return digits.data, digits.target
