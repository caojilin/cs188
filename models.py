import nn
import numpy as np
import backend


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        score = nn.as_scalar(self.run(x))
        if score >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        for i in range(50):
            batch_size = 1
            for x, y in dataset.iterate_once(batch_size):
                y_star = self.get_prediction(x)
                if y_star != nn.as_scalar(y):
                    self.w.update(x, -y_star)
            count = 0
            for x, y in dataset.iterate_once(batch_size):
                y_star = self.get_prediction(x)
                if y_star != nn.as_scalar(y):
                    count += 1
            if count == 0:
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.w1 = nn.Parameter(1, 128)
        self.w2 = nn.Parameter(128, 1)
        self.b1 = nn.Parameter(1, 128)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        H1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        predicted_y = nn.AddBias(nn.Linear(H1, self.w2), self.b2)
        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        loss = nn.SquareLoss(self.run(x), y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        w1, w2, b1, b2 = self.w1, self.w2, self.b1, self.b2
        x = nn.Constant(dataset.x)
        y = nn.Constant(dataset.y)
        epoch = 4000
        num_epoch = 0
        for episode in range(epoch):
            head = 0
            batch_size = 20
            for i in range(10):
                inp_x = nn.Constant(dataset.x[head:batch_size])
                inp_y = nn.Constant(dataset.y[head:batch_size])
                loss = self.get_loss(inp_x, inp_y)
                grad_wrt_w1, grad_wrt_w2, grad_wrt_b1, grad_wrt_b2 = nn.gradients(loss, [w1, w2, b1, b2])
                self.w1.update(grad_wrt_w1, -0.01)
                self.w2.update(grad_wrt_w2, -0.01)
                self.b1.update(grad_wrt_b1, -0.01)
                self.b2.update(grad_wrt_b2, -0.01)
                head += 20
                batch_size += 20
            num_epoch += 1
            predicted_y = self.run(x)
            training_loss = nn.as_scalar(self.get_loss(predicted_y, y))
            if training_loss < 0.02:
                break


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.w1 = nn.Parameter(784, 128)
        self.w2 = nn.Parameter(128, 10)
        self.b1 = nn.Parameter(1, 128)
        self.b2 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        H1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        predicted_y = nn.AddBias(nn.Linear(H1, self.w2), self.b2)
        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        loss = nn.SoftmaxLoss(self.run(x), y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        w1, w2, b1, b2 = self.w1, self.w2, self.b1, self.b2
        x = nn.Constant(dataset.x)
        print(len(x.data))
        epoch = 20
        num_epoch = 0
        learning_rate = -1
        accuracy = 0
        for episode in range(epoch):
            head = 0
            batch_size = 600
            for i in range(100):
                inp_x = nn.Constant(dataset.x[head:batch_size])
                inp_y = nn.Constant(dataset.y[head:batch_size])
                loss = self.get_loss(inp_x, inp_y)
                grad_wrt_w1, grad_wrt_w2, grad_wrt_b1, grad_wrt_b2 = nn.gradients(loss, [w1, w2, b1, b2])
                self.w1.update(grad_wrt_w1, learning_rate)
                self.w2.update(grad_wrt_w2, learning_rate)
                self.b1.update(grad_wrt_b1, learning_rate)
                self.b2.update(grad_wrt_b2, learning_rate)
                head += 600
                batch_size += 600
            if accuracy > 0.96:
                learning_rate = -0.4
            elif accuracy >= 0.97:
                learning_rate = -0.2
            num_epoch += 1
            accuracy = dataset.get_validation_accuracy()
            print(accuracy, "epoch=", num_epoch)
            # if accuracy > 0.97:
            #     break


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.05
        self.hidden_size = 275

        self.w1 = nn.Parameter(self.num_chars, self.hidden_size)
        self.w2 = nn.Parameter(self.hidden_size, 5)
        self.b2 = nn.Parameter(1, 5)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        batch_size = len(xs[0].data)
        # print("xs[0].data", xs[0].data)
        # print("batch size", batch_size)
        self.h_vec = nn.Constant(np.zeros([batch_size, self.hidden_size]))

        for chars in xs:

            mm = nn.Linear(chars, self.w1)
            # keep updating h_vec
            self.h_vec = nn.ReLU(nn.Add(mm, self.h_vec))

        another_mm = nn.Linear(self.h_vec, self.w2)
        self.h_final = nn.AddBias(another_mm, self.b2)

        # print(self.h_final)
        return self.h_final

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        loss = nn.SoftmaxLoss(self.run(xs), y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        epoch = 100
        batch_size = 100
        learning_rate = -1
        num_epoch = 0
        for episode in range(epoch):
            for inp_x, inp_y in dataset.iterate_once(batch_size):
                loss = self.get_loss(inp_x, inp_y)
                grad_wrt_w1, grad_wrt_w2, grad_wrt_b2 = nn.gradients(loss, [self.w1, self.w2, self.b2])
                self.w1.update(grad_wrt_w1, learning_rate)
                self.w2.update(grad_wrt_w2, learning_rate)
                self.b2.update(grad_wrt_b2, learning_rate)
            num_epoch += 1
            accuracy = dataset.get_validation_accuracy()
            print(accuracy, "epoch=", num_epoch)
            # if accuracy > 0.81:
            #     return


# linear regression demo
def linearDemo():
    x = nn.Constant(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float))
    y = np.transpose(np.array([[3, 11, 10, 18]], dtype=np.float))
    y = nn.Constant(y)
    m = nn.Parameter(2, 1)
    b = nn.Parameter(1, 1)
    xm = nn.Linear(x, m)
    predicted_y = nn.AddBias(xm, b)
    loss = nn.SquareLoss(predicted_y, y)
    grad_wrt_m, grad_wrt_b = nn.gradients(loss, [m, b])
    m.update(grad_wrt_m, -0.5)
    b.update(grad_wrt_b, -0.5)

    while loss.data > 0.001:
        xm = nn.Linear(x, m)
        predicted_y = nn.AddBias(xm, b)
        loss = nn.SquareLoss(predicted_y, y)
        grad_wrt_m, grad_wrt_b = nn.gradients(loss, [m, b])
        m.update(grad_wrt_m, -0.5)
        b.update(grad_wrt_b, -0.5)
        print(loss.data)


def test():
    model = RegressionModel()
    dat = backend.RegressionDataset(model)
    batch_size = 200
    w1 = nn.Parameter(128, batch_size)
    w2 = nn.Parameter(batch_size, 128)
    b1 = nn.Parameter(1, 1)
    b2 = nn.Parameter(1, 1)
    x = nn.Constant(dat.x[:batch_size])
    y = nn.Constant(dat.y[:batch_size])

    epoch = 100
    for episode in range(epoch):
        H1 = nn.ReLU(nn.AddBias(nn.Linear(w1, x), b1))
        predicted_y = nn.AddBias(nn.Linear(w2, H1), b2)
        loss = nn.SquareLoss(predicted_y, y)
        grad_wrt_w1, grad_wrt_w2, grad_wrt_b1, grad_wrt_b2 = nn.gradients(loss, [w1, w2, b1, b2])
        w1.update(grad_wrt_w1, -0.1)
        w2.update(grad_wrt_w2, -0.1)
        b1.update(grad_wrt_b1, -0.1)
        b2.update(grad_wrt_b2, -0.1)
        training_loss = nn.as_scalar(loss)
        if training_loss < 0.02:
            break

    # a = np.array([[1, 2], [2, 4]], dtype=np.float)
    # b = np.array([2, 3], dtype=np.float)
    # np.matmul(a,b)
