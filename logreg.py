import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    newRegression = LogisticRegression()
    newRegression.fit(x_train, y_train)
    pred_result = newRegression.predict(x_valid)
    np.savetxt(save_path, pred_result)
    return pred_result


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
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
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        def initiate_theta(dim):
            self.theta = np.zeros(dim)
            # print('self.theta initiated is {}'.format(self.theta))
            
        def implement_sigmoid(x):
            if self.theta is None:
                initiate_theta(x.shape[1])
            z = np.matmul(np.transpose(self.theta), np.transpose(x))
            return 1/(np.ones(x.shape[0]) + np.exp(-z))
    
        def implement_partial_loss(x, y):
            return -np.matmul(np.transpose(y - implement_sigmoid(x)), x)/x.shape[0]
    
        def implement_transposed_hess(x):
            sigmoid_hadamard = implement_sigmoid(x) * (np.ones(x.shape[0]) - implement_sigmoid(x))
            hess2 = np.diag(sigmoid_hadamard)
            hess = np.matmul(hess2,x)
            hess = np.matmul(np.transpose(x),hess)/x.shape[0]
            hess_inverse = np.linalg.inv(hess)
            return hess_inverse
        
        def train(x, y):
            count = 0
            if self.theta is None:
                initiate_theta(x.shape[1])
            while count < self.max_iter:
                if self.verbose:
                    loss_y1 = np.matmul(np.transpose(y), np.log(implement_sigmoid(x)))
                    loss_y0 = np.matmul(np.transpose(np.ones(x.shape[0]) - y), np.log(np.ones(x.shape[0]) - implement_sigmoid(x)))
                    loss = -(loss_y1 + loss_y0 )/x.shape[0]
                    print('Average empirical loss for step {} is {}'.format(count, loss))
                delta = np.matmul(implement_transposed_hess(x), implement_partial_loss(x, y))
                new_theta = self.theta - delta * self.step_size
                delta_theta = np.linalg.norm(new_theta - self.theta)
                # print('delta is {}'.format(delta_theta))
                if delta_theta < self.eps:
                    return new_theta
                else:
                    self.theta = new_theta
                count += 1
            return self.theta
        
        return train(x, y)
    
    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        if self.theta is None:
            initiate_theta(x.shape[1])
        z = np.matmul(np.transpose(self.theta), np.transpose(x))
        sigmoid_output = 1/(np.ones(x.shape[0]) + np.exp(-z))
        return np.where(sigmoid_output>=0.5, 1, 0)

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
