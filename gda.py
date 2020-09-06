import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=False)
    
    newRegression = GDA()
    newRegression.fit(x_train, y_train)
    pred_result = newRegression.predict(x_valid)
    np.savetxt(save_path, pred_result)
    return pred_result


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
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
        self.params = None

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        
        def calculate_phi(y):
            return np.mean(y)
        
        def calculate_mu0(x,y):
            zeros_2_ones = np.where(y == 0, 1, 0)
            return np.matmul(np.transpose(zeros_2_ones), x)/np.sum(zeros_2_ones)
        
        def calculate_mu1(x,y):
            return np.matmul(np.transpose(y), x)/np.sum(y)
        
        def calculate_sigma(x,y,mu0, mu1):
            zeros_2_ones = np.where(y == 0, 1, 0)
            result = sum([y[i]*np.outer((x[i,:] - mu1),(x[i,:] - mu1)) +  zeros_2_ones[i]*np.outer((x[i,:] - mu0),(x[i,:] - mu0)) for i in range(x.shape[0])])
            return result/x.shape[0]         
        
        phi = calculate_phi(y)
        mu0 = calculate_mu0(x,y)
        mu1 = calculate_mu1(x,y)
        sigma = calculate_sigma(x, y, mu0, mu1)
        sigma_inv = np.linalg.inv(sigma)
        theta_base = np.matmul(np.transpose(sigma_inv),(mu1 - mu0))
        mu0_sq = np.matmul(np.transpose(mu0),np.matmul(sigma_inv, mu0))
        mu1_sq = np.matmul(np.transpose(mu1),np.matmul(sigma_inv, mu1))
        theta_const = np.array([0.5*(mu0_sq - mu1_sq) - np.log((1-phi)/phi)])
        self.theta = np.concatenate([theta_const, theta_base])
        self.params = (phi, mu0, mu1, sigma)
    
    #prob_of_x_given_yo(phi, mu0, sigma, x_i), prob_of_x_given_y1(phi, mu1, sigma, x_i)
    
    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """

        def prob_of_x_given_yo(mu0, sigma, x_i):
            diff = x_i - mu0
            temp = np.matmul(np.linalg.inv(sigma), diff)
            temp = np.matmul(np.transpose(diff), temp)
            temp_exp = np.exp(-0.5*temp)
            pi_power = np.power((2*np.pi),len(x_i))
            abs_sigma_power = np.power(np.linalg.det(sigma), 0.5)
            return (1/(pi_power*abs_sigma_power))*temp_exp
        
        def prob_of_x_given_y1(mu1, sigma, x_i):
            diff = x_i - mu1
            temp = np.matmul(np.linalg.inv(sigma), diff)
            temp = np.matmul(np.transpose(diff), temp)
            temp_exp = np.exp(-0.5*temp)
            pi_power = np.power((2*np.pi),len(x_i))
            abs_sigma_power = np.power(np.linalg.det(sigma), 0.5)
            return (1/(pi_power*abs_sigma_power))*temp_exp
        
        def prob_of_y0_given_x(x_yo, x_y1, phi):
            denominator = x_yo + x_y1
            return (x_yo*(1-phi))/denominator
        
        def prob_of_y1_given_x(x_yo, x_y1, phi):
            denominator = x_yo + x_y1
            return (x_y1*phi)/denominator
       
        phi, mu0, mu1, sigma = self.params
        list_r = []
        for i in range(x.shape[0]):
            x_yo = prob_of_x_given_yo(mu0, sigma, x[i,:])
            x_y1 = prob_of_x_given_y1(mu1, sigma, x[i,:])
            y0_given_x = prob_of_y0_given_x(x_yo, x_y1, phi)
            y1_given_x = prob_of_y1_given_x(x_yo, x_y1, phi)
            if max(y0_given_x, y1_given_x) == y0_given_x:
                list_r.append(0)
            else:
                list_r.append(1)
        return np.array(list_r)

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
