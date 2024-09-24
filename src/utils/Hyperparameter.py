import numpy as np

from itertools import product
from minisom import MiniSom

class HyperparameterTuning:
    """
    The class is used to tune the hyperparameters of the SOM model.
    
    It is based on the MiniSom library and performs a grid search over the
    sigma and learning rate hyperparameters.
     
    The best hyperparameters are selected based on the quantization error of the model.
    
    :return: 
    Dictionary containing the best hyperparameters found.
    """
    def __init__(self, data:np.ndarray, n_neurons:int=None, pca=True):
        self.data = data
        self.sigma_list = np.arange(0.1, 6, 0.3)
        self.learning_rate_list = np.arange(0.1, 6, 0.3)
        self.best_quant_error = float('inf')
        self.random_seed = 0
        self.neighborhood_function = 'gaussian'
        self.topology = 'rectangular'
        self.pca = pca
        self.best_params = {}

        if n_neurons is None:
            self.n_neurons = np.sqrt(5 * self.data.shape[0]).astype(int)
        else:
            self.n_neurons = n_neurons

    def run(self):
        for sigma, learning_rate in product(self.sigma_list, self.learning_rate_list):
            som = MiniSom(
                self.n_neurons,
                self.n_neurons,
                self.data.shape[1],
                sigma=sigma,
                learning_rate=learning_rate,
                neighborhood_function='gaussian',
                random_seed=0,
                topology='rectangular'
            )
            if self.pca:
                som.pca_weights_init(self.data)
            else:
                som.random_weights_init(self.data)

            som.train(self.data, 1000, verbose=False)

            quant_error = som.quantization_error(self.data)

            if quant_error < self.best_quant_error:
                self.best_quant_error = quant_error
                self.best_params['sigma'] = sigma
                self.best_params['learning_rate'] = learning_rate

                print('Best params so far: Sigma{}, Learning Rate{}, Quant Error{}'.format(
                    sigma, learning_rate, quant_error
                ))

        print('Best hyperparameters found: Sigma{},'
              ' Learning Rate{}, Quant Error{}'.format(
            self.best_params['sigma'],
            self.best_params['learning_rate'],
            self.best_quant_error
        ))

        self.best_params['n_neurons'] = self.n_neurons
        self.best_params['random_seed'] = self.random_seed
        self.best_params['neighborhood_function'] = self.neighborhood_function
        self.best_params['topology'] = self.topology

        return self.best_params

    def convert_to_native_types(self):
        converted_params = {}
        for key, value in self.best_params.items():
            if isinstance(value, np.generic):
                converted_params[key] = value.item()
            else:
                converted_params[key] = value
        return converted_params