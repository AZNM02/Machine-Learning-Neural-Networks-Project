import nn

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
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        score = nn.as_scalar(self.run(x))
        return 1 if score >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        batch_size = 1

        while True:
            mistakes = 0

            for x, y in dataset.iterate_once(batch_size):
                # y is a Constant node with shape (1 x 1)
                true_label = nn.as_scalar(y)  # +1 or -1
                pred_label = self.get_prediction(x)

                if pred_label != true_label:
                    mistakes += 1
                    # direction has same shape as w: (1 x dimensions)
                    # x has shape (1 x dimensions); y has shape (1 x 1)
                    # broadcasting y * x  - > (1 x dimensions)
                    direction = nn.Constant(y.data * x.data)
                    # learning rate = 1.0 so update is exactly w += y * x
                    self.w.update(direction, 1.0)

            if mistakes == 0:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Architecture: 1 -> 256 -> 256 -> 256 -> 1
        self.h = 256
        self.lr = 0.01
        self.batch = 200

        self.w1 = nn.Parameter(1, self.h)
        self.b1 = nn.Parameter(1, self.h)

        self.w2 = nn.Parameter(self.h, self.h)
        self.b2 = nn.Parameter(1, self.h)

        self.w3 = nn.Parameter(self.h, self.h)
        self.b3 = nn.Parameter(1, self.h)

        self.w4 = nn.Parameter(self.h, 1)
        self.b4 = nn.Parameter(1, 1)

        self.parameters = [
            self.w1, self.b1,
            self.w2, self.b2,
            self.w3, self.b3,
            self.w4, self.b4
        ]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        # Layer 1
        h1 = nn.Linear(x, self.w1)
        h1 = nn.AddBias(h1, self.b1)
        h1 = nn.ReLU(h1)

        # Layer 2
        h2 = nn.Linear(h1, self.w2)
        h2 = nn.AddBias(h2, self.b2)
        h2 = nn.ReLU(h2)

        # Layer 3
        h3 = nn.Linear(h2, self.w3)
        h3 = nn.AddBias(h3, self.b3)
        h3 = nn.ReLU(h3)

        # Output layer
        y = nn.Linear(h3, self.w4)
        y = nn.AddBias(y, self.b4)
        return y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model using gradient descent until the loss is small enough
        or a maximum number of epochs is reached.
        """
        max_epochs = 6000
        target = 0.02

        for epoch in range(max_epochs):
            for x_batch, y_batch in dataset.iterate_once(self.batch):
                loss = self.get_loss(x_batch, y_batch)
                grads = nn.gradients(loss, self.parameters)

                # Gradient descent update
                for param, g in zip(self.parameters, grads):
                    param.update(g, -self.lr)

            # Check training loss after each epoch
            full_loss = nn.as_scalar(self.get_loss(
                nn.Constant(dataset.x),
                nn.Constant(dataset.y)
            ))

            # Optional: print progress
            # print("Epoch", epoch, "Loss:", full_loss)

            if full_loss <= target:
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
        # Architecture hyperparameters
        self.input_dim = 28 * 28      # 784
        self.hidden1 = 200
        self.hidden2 = 100
        self.num_classes = 10

        # Optimisation hyperparameters
        self.learning_rate = 0.2
        self.batch_size = 100
        # We’ll stop when validation accuracy is good enough
        self.target_val_acc = 0.975

        # Parameters for a 2-hidden-layer MLP:
        # 784 -> 200 -> 100 -> 10
        self.w1 = nn.Parameter(self.input_dim, self.hidden1)
        self.b1 = nn.Parameter(1, self.hidden1)

        self.w2 = nn.Parameter(self.hidden1, self.hidden2)
        self.b2 = nn.Parameter(1, self.hidden2)

        self.w3 = nn.Parameter(self.hidden2, self.num_classes)
        self.b3 = nn.Parameter(1, self.num_classes)

        self.parameters = [
            self.w1, self.b1,
            self.w2, self.b2,
            self.w3, self.b3,
        ]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing logits
        """
        # First hidden layer: Linear + bias + ReLU
        h1 = nn.Linear(x, self.w1)       
        h1 = nn.AddBias(h1, self.b1)
        h1 = nn.ReLU(h1)

        # Second hidden layer: Linear + bias + ReLU
        h2 = nn.Linear(h1, self.w2)       
        h2 = nn.AddBias(h2, self.b2)
        h2 = nn.ReLU(h2)

        # Output layer: Linear + bias (NO ReLU — we want raw logits)
        logits = nn.Linear(h2, self.w3)   
        logits = nn.AddBias(logits, self.b3)
        return logits

    def get_loss(self, x, y):
        """
        Computes the softmax loss for a batch of examples.

        Inputs:
            x: node with shape (batch_size x 784)
            y: node with shape (batch_size x 10), one-hot labels
        Returns:
            A loss node (scalar)
        """
        logits = self.run(x)
        return nn.SoftmaxLoss(logits, y)

    def train(self, dataset):
        """
        Trains the model using minibatch gradient descent.

        Uses validation accuracy to decide when to stop training.
        """
        max_epochs = 30

        for epoch in range(max_epochs):
            # One full pass over training data
            for x_batch, y_batch in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x_batch, y_batch)
                grads = nn.gradients(loss, self.parameters)

                # Gradient descent update: param <- param - lr * grad
                for param, grad in zip(self.parameters, grads):
                    param.update(grad, -self.learning_rate)

            # Check validation accuracy after each epoch
            val_acc = dataset.get_validation_accuracy()
            # print("Epoch", epoch, "validation accuracy:", val_acc)  # (optional debug)

            if val_acc >= self.target_val_acc:
                break

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
        # Hyperparameters 
        self.hidden_size = 128          
        self.learning_rate = 0.1
        self.batch_size = 100
        self.target_val_acc = 0.82      

        num_langs = len(self.languages)

        # RNN parameters 
        # Input-to-hidden
        self.Wx = nn.Parameter(self.num_chars, self.hidden_size)
        # Hidden-to-hidden
        self.Wh = nn.Parameter(self.hidden_size, self.hidden_size)
        # Hidden bias
        self.bh = nn.Parameter(1, self.hidden_size)

        # Output (hidden -> language logits)
        self.Wo = nn.Parameter(self.hidden_size, num_langs)
        self.bo = nn.Parameter(1, num_langs)

        self.parameters = [self.Wx, self.Wh, self.bh, self.Wo, self.bo]

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
        h = None
        for t, x_t in enumerate(xs):
            if h is None:
                # First character: only depends on x_0
                z = nn.Linear(x_t, self.Wx)
            else:
                # Later characters: combine current x_t and previous h
                z_x = nn.Linear(x_t, self.Wx)
                z_h = nn.Linear(h, self.Wh)
                z = nn.Add(z_x, z_h)
            z = nn.AddBias(z, self.bh)
            h = nn.ReLU(z)

        # After processing the whole word, map final hidden state to logits
        logits = nn.Linear(h, self.Wo)
        logits = nn.AddBias(logits, self.bo)
        return logits

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
        logits = self.run(xs)
        return nn.SoftmaxLoss(logits, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        max_epochs = 50

        for epoch in range(max_epochs):
            # One epoch over randomized buckets / batches
            for xs, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(xs, y)
                grads = nn.gradients(loss, self.parameters)

                # Gradient descent updates
                for param, grad in zip(self.parameters, grads):
                    param.update(grad, -self.learning_rate)

            # Check dev accuracy after each epoch
            val_acc = dataset.get_validation_accuracy()
            # print("Epoch", epoch, "validation accuracy:", val_acc)  # optional

            if val_acc >= self.target_val_acc:
                break
