import numpy as np
import util
import matplotlib.pyplot as plt

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    newPoisson = PoissonRegression()
    newPoisson.fit(x_train, y_train)
    pred_result = newPoisson.predict(x_eval)
    np.savetxt(save_path, pred_result)
    return pred_result


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        def initiate_theta(dim):
            """Initiate theta as a 1 dimensional vector with dim same as x.shape[1] - number of x features
            
            Args:
                dim: The number of features plus 1. This is the Width of x i.e dim of Shape(n_examples, dim)
            """
            self.theta = np.zeros(dim)
            
            
        def implement_hypothesis(x):
            """Calculate the exponential of eta [(theta)^T.x]. 
            
            Args: 
                x: Training example inputs. Shape (n_examples, dim).
                
            Output: 
                hypothesis: A vector that contains the exp of theta^T.x or hypothesis for each example. Shape(n_examples,).
            """
            if self.theta is None:
                initiate_theta(x.shape[1])
            z = np.matmul(np.transpose(self.theta), np.transpose(x))
            return np.exp(z)
        
        def implement_partial_loss(x, y):
            """Calculate the vector for the partial derivative of the loss function (batch). Calculates the sum over all
                n_examples for each dim
            
            Args:
                x: Training example inputs. Shape (n_examples, dim).
                y: Training example labels. Shape (n_examples,). 
            
            Output: 
                Vector of Shape (dim,) containing the partial derivative of the loss function for theta of each dim
            """
            return np.matmul(np.transpose(y - implement_hypothesis(x)), x)
        
        
        def train(x, y):
            """Trains the regression using batch gradient ascent to maximize the log-likelihood of theta.
            
            Args:
                x: Training example inputs. Shape (n_examples, dim).
                y: Training example labels. Shape (n_examples,). 
                
            Output: 
                Final theta derived from the training after achieving maximum iterations allowed or reaching
                improvement limit self.eps.
            """
            count = 0
            if self.theta is None:
                initiate_theta(x.shape[1])
            while count < self.max_iter:
                if self.verbose:
                    loss_term_1 = sum(implement_hypothesis(x))
                    loss_term_2 = np.matmul(np.transpose(y), np.log(implement_hypothesis(x)))
                    loss = (loss_term_1 - loss_term_2 )/x.shape[0]
                    print('Average empirical loss for step {} is {}'.format(count, loss))
                new_theta = self.theta + implement_partial_loss(x, y) * self.step_size
                delta_theta = np.linalg.norm(new_theta - self.theta)
                print('delta is {}'.format(delta_theta))
                if delta_theta < self.eps:
                    return new_theta
                else:
                    self.theta = new_theta
                count += 1
            return self.theta
        
        return train(x, y)

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        if self.theta is None:
            initiate_theta(x.shape[1])
        z = np.matmul(np.transpose(self.theta), np.transpose(x))
        return np.exp(z)

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
