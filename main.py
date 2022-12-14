import os
import numpy as np
import matplotlib.pyplot as plt


input_dim = 128
hidden_dim = 64
data_dim = 64
sample_size = 64
step = 128
repeat = 1
config_str = "input_dim=" + str(input_dim) + \
            "hidden_dim=" + str(hidden_dim) + \
            "data_dim=" + str(data_dim) + \
            "sample_size=" + str(sample_size) + \
            "training_step=" + str(step) + \
            "repeat=" + str(repeat)
plot_path = os.getcwd() + "/results/"
os.makedirs(plot_path, exist_ok=True)


def relu(x):
    return (np.abs(x)+x)/2


def sigmoid(x):
    return 1/(1+np.exp(-x))


def indicator_nonnegative(x):
    if x>=0:
        return 1
    else:
        return 0
    

def plot_curve_line(x, y, std, color, label, width, alpha):
    plt.plot(x, y, c=color, label=label, linewidth=width)
    plt.fill_between(x, y-std, y+std, color=color, alpha=alpha)


class two_layer_relu_nn():
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W = np.zeros((self.hidden_dim, self.input_dim))
        self.a = np.zeros((1, self.hidden_dim))
        self.relu = relu
        
    def forward(self, x):
        x = np.dot(self.W, x)
        x = self.relu(x)
        x = np.dot(self.a, x)
        return x / np.sqrt(self.hidden_dim)
    
    def init_gaussian(self):
        self.a = (np.random.binomial(1, 0.5, (1, self.hidden_dim))-0.5)*2
        for i in range(self.hidden_dim):
            for j in range(self.input_dim):
                self.W[i, j] = np.random.normal(loc=0, scale=1.0)

    def init_parallel(self, x):
        assert x.shape[1] == self.hidden_dim
        self.a = (np.random.binomial(1, 0.5, (1, self.hidden_dim))-0.5)*2
        for i in range(self.hidden_dim):
            self.W[i,:] = x[:, i] * np.random.normal(loc=0, scale=1)
    
    def init_orthogonal(self, weight_base):
        weight_coordindate = np.random.normal(loc=0, scale=1.0, size=(self.hidden_dim, self.hidden_dim))   
        self.W = np.dot(weight_coordindate, weight_base)
        self.a = (np.random.binomial(1, 0.5, (1, self.hidden_dim))-0.5)*2

    def mse_loss(self, y, predict):
        return np.sum((y-predict)*(y-predict)) / (2*y.shape[1])

    def train_gd(self, x, y, lr, T, repeat_count=0, loss_array=None):

        output = self.forward(x)
        print('initial loss:',self.mse_loss(y,output))

        for epoch in range(T):
            #print('ntk min eigenvalue:', self.ntk_min_eigen(x, 0.0001))
            output = self.forward(x)
            loss_last = self.mse_loss(y, output)
            for r in range(self.hidden_dim):
                grad = np.zeros((self.input_dim,))
                for i in range(sample_size):
                    grad += (output[0][i]-y[0][i]) * self.a[0][r] * x[:,i] * indicator_nonnegative(np.dot(self.W[r,:],x[:,i]).squeeze())
                grad = grad / np.sqrt(self.hidden_dim)
                self.W[r,:] += -lr * grad

            output = self.forward(x)
            loss = self.mse_loss(y, output)
            print('epoch:', epoch, 'loss:', loss, 'ratio:', loss / loss_last)
            loss_array[repeat_count][epoch] = loss

    def ntk(self, x, l=0.0001):
        H = np.zeros((sample_size, sample_size))
        for i in range(sample_size):
            for j in range(sample_size):
                for r in range(hidden_dim):
                    H[i, j] += indicator_nonnegative(np.dot(self.W[r,:],x[:,i]).squeeze()) * indicator_nonnegative(np.dot(self.W[r,:],x[:,j]).squeeze())
                H[i, j] *= np.dot(x[:, i], x[:, j])
                H[i, j] /= hidden_dim
        H += l * np.identity(sample_size)
        return H

    def ntk_min_eigen(self, x, l=0.0001):
        H = self.ntk(x, l)
        eig_value, eig_vector = np.linalg.eig(H)
        index = np.argmin(eig_value)
        return eig_value[index]


class two_layer_sigmoid_nn():
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W = np.ones((self.hidden_dim, self.input_dim))
        self.a = (np.random.binomial(1, 0.5, (1, self.hidden_dim))-0.5)*2
        # print(self.a)

        
    def forward(self, x):
        x = np.dot(self.W, x)
        x = sigmoid(x)
        x = np.dot(self.a, x)
        return x / np.sqrt(self.hidden_dim)


    def init_parallel(self, x):
        assert x.shape[1] == self.hidden_dim
        for i in range(self.hidden_dim):
            self.W[i,:] = x[:, i]*np.random.normal(loc=0, scale=1.0)

    
    def init_orthogonal(self, weight_base):
        weight_coordindate = np.random.normal(loc=0, scale=1.0, size=(self.hidden_dim, input_dim-sample_size))   
        self.W = np.dot(weight_coordindate, weight_base)

    
    def init_all_parallel(self, x):
        assert x.shape[1] == self.hidden_dim
        for i in range(self.hidden_dim):
            self.W[i,:] = x[:, i]*np.random.normal(loc=0, scale=1.0)
        self.a = (np.random.binomial(1, 0.5, (1, self.hidden_dim))-0.5)*2

    
    def init_all_orthogonal(self, weight_base):
        weight_coordindate = np.random.normal(loc=0, scale=1.0, size=(self.hidden_dim, input_dim-sample_size))   
        self.W = np.dot(weight_coordindate, weight_base)
        self.a = (np.random.binomial(1, 0.5, (1, self.hidden_dim))-0.5)*2


    def mse_loss(self, y, predict):
        return np.sum((y-predict)*(y-predict)) / (2*sample_size)


    def train_gd(self, x, y, lr, T, repeat_count=0, loss_array=None):
        output = model.forward(x)
        print('initial loss:',self.mse_loss(y,output))
        for epoch in range(T):
            output = model.forward(x)
            for r in range(self.hidden_dim):
                grad = np.zeros((self.input_dim,))
                for i in range(sample_size):
                    grad += (output[0][i]-y[0][i])*self.a[0][r]*x[:,i]*sigmoid(np.dot(self.W[r,:],x[:,i]))*(1-sigmoid(np.dot(self.W[r,:],x[:,i])))
                grad = grad / np.sqrt(self.hidden_dim)
                self.W[r,:] += -lr * grad
            output = model.forward(x)
            print('epoch:', epoch,'loss:', self.mse_loss(y,output))
            loss_array[repeat_count][epoch] = self.mse_loss(y,output)


if __name__ == '__main__':
    
    # orthogonal initialization data
    assert data_dim + hidden_dim <= input_dim
    base = np.identity(input_dim)
    x_orth_base = base[:, hidden_dim:hidden_dim + data_dim]
    weight_base = np.transpose(base[:,0:hidden_dim])
    x_orth_coordinate = np.random.normal(size=(data_dim, sample_size), scale=1.0)
    x_orth = np.dot(x_orth_base, x_orth_coordinate)
    for i in range(sample_size):
        x_orth[:,i] = x_orth[:,i]/np.linalg.norm(x_orth[:,i], ord=2)
    y = (np.random.binomial(1, 0.5, size=(1,sample_size))-0.5) * 2

    # model
    model = two_layer_relu_nn(input_dim, hidden_dim)

    loss_gaussian = np.zeros((repeat, step))
    loss_parallel = np.zeros((repeat, step))
    loss_orthogonal = np.zeros((repeat, step))
    loss_gaussian_mean, loss_parallel_mean, loss_orthogonal_mean, loss_gaussian_std, loss_parallel_std, loss_orthogonal_std = [], [], [], [], [], []

    for i in range(repeat):
        print("====== repeat ", i + 1, " ======")

        print("====== gaussian ======")
        model.init_gaussian()
        model.train_gd(x_orth, y, 0.1, step, i, loss_gaussian)

        print("====== parallel ======")
        model.init_parallel(x_orth)
        model.train_gd(x_orth, y, 0.1, step, i, loss_parallel)

        print("====== orthogonal ======")
        model.init_orthogonal(weight_base)
        model.train_gd(x_orth, y, 0.1, step, i, loss_orthogonal)
        
    for i in range(step):
        loss_parallel_mean.append(np.mean(loss_parallel[:,i]))
        loss_parallel_std.append(np.std(loss_parallel[:,i]))
        loss_gaussian_mean.append(np.mean(loss_gaussian[:,i]))
        loss_gaussian_std.append(np.std(loss_gaussian[:,i]))
        loss_orthogonal_mean.append(np.mean(loss_orthogonal[:,i]))
        loss_orthogonal_std.append(np.std(loss_orthogonal[:,i]))
    
    loss_parallel_mean = np.array(loss_parallel_mean)
    loss_gaussian_mean = np.array(loss_gaussian_mean)
    loss_orthogonal_mean = np.array(loss_orthogonal_mean)
    loss_parallel_std = np.array(loss_parallel_std)
    loss_gaussian_std = np.array(loss_gaussian_std)
    loss_orthogonal_std = np.array(loss_orthogonal_std)


    x_step = np.linspace(1,step,step)
    plot_curve_line(x_step, loss_parallel_mean, loss_parallel_std, 'blue', 'parallel initialization', width=1.0, alpha=0.2)
    plot_curve_line(x_step, loss_gaussian_mean, loss_gaussian_std, 'green', 'gaussian initialization', width=1.0, alpha=0.2)
    plot_curve_line(x_step, loss_orthogonal_mean, loss_orthogonal_std, 'orange', 'orthogonal initialization', width=1.0, alpha=0.2)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title(config_str)
    plt.legend(loc='best')
    plt.savefig(plot_path + config_str, dpi=400)
    plt.show()
