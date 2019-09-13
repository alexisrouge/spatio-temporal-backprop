import numpy as np

tau = .1  # capacitor time constant
V_th = 1.5  # threshold potential
a = 1 # slope factor for dirac function derivative estimate (gradS)


def initW(size_in, size_out):
    # initialization function for network weights. Recommended form of Wu et al.
    W = 2*np.random.random((size_out, size_in)) - 1
    sqrW = W**2
    normW = np.sqrt(np.sum(sqrW, axis=1)).reshape((size_out, 1))
    return W / normW


def forget(x):
    # capacitor discharge function
    return tau*np.exp(-x/tau)


def gradF(x):
    # gradient of previous function
    return -np.exp(-x/tau)


def spiker(x):
    # dirac delta function, emits 1 if x greater or equal to threshold potential
    return np.array(x >= V_th, np.int)


def gradS(x):
    # approximate derivative of dirac delta function
    return 1/(np.sqrt(2*np.pi*a))*np.exp(-(x-V_th)**2/(2*a))


def ber(p):
    # helper function to create input spike train
    return np.array(np.random.random(p.shape) <= p, np.int)


class LIF_layer:
    # class implementing a single layer of size_out neurons.
    def __init__(self, size_in, size_out, forget, spiker, gradF, gradS):
        self.W = initW(size_in, size_out)
        self.bias = np.random.random(size_out)*1.5
        self.forget = forget
        self.gradF = gradF
        self.gradS = gradS
        self.spiker = spiker
        self.size_out = size_out
        self.X = np.zeros(size_out)
        self.U = np.zeros(size_out)
        self.Out = np.zeros(size_out)
        self.spikes = []
        self.potential = []

    def reset(self):
        # reset layer variables and history ton 0
        self.X = np.zeros(self.size_out)
        self.U = np.zeros(self.size_out)
        self.Out = np.zeros(self.size_out)
        self.spikes = []
        self.potential = []

    def run(self, inpt, hist=False):
        # run one step of the neuron dynamic and store values for
        # backprop if needed
        self.X = np.dot(inpt, self.W.T)
        self.U = self.U*self.forget(self.Out) + self.X + self.bias
        self.Out = self.spiker(self.U)
        if hist:
            self.spikes.append(self.Out)
            self.potential.append(self.U)
        return self.Out

    def grad(self, last, border, t, target, W, deltaN, deltaT, epsilonT, UN):
        # compute gradient of loss function with respect to U and O
        # The 4 different cases of the paper are reused here
        # last == True means t=T and border == True means n=N
        # delta = dL/do(t,n), epsilon = dL/du(t,n)
        # target is the one hot encoded vector of labels
        # W is the weight vector of the layer n+1
        # deltaN = dL/do(t, n+1), deltaT = dL/do(t+1, n)
        # epsilonT = dL/du(t+1, n), UN = u(t, n+1)
        if last:
            if border:
                S = self.spikes[0].shape[0]  # number of samples
                T = len(self.spikes)  # number of time step
                delta = -1/T/S*(target - 1/T*np.sum(self.spikes, axis=0))
            else:
                delta = np.dot(deltaN*self.gradS(UN), W)
        else:
            delta = deltaT * self.potential[t] * self.gradS(self.potential[t+1]) * self.gradF(self.spikes[t])
            if border:
                S = self.spikes[0].shape[0]  # number of samples
                T = len(self.spikes)  # number of time step
                delta = delta - 1/T/S*(target - 1/T*np.sum(self.spikes, axis=0))
            else:
                delta = delta + np.dot(deltaN * self.gradS(UN), W)

        epsilon = delta * self.gradS(self.potential[t])
        if not last:
            epsilon = epsilon + epsilonT * self.forget(self.spikes[t])
        return delta, epsilon


class LIF_net:
    # implement the hole network, embbeding all the layers
    # T here is the number of time step during which the input is exposed to
    # the network
    def __init__(self,
                 size_in=784,
                 hidden_size=800,
                 size_out=10,
                 forget=forget,
                 spiker=spiker,
                 gradF=gradF,
                 gradS=gradS,
                 T=30):
        self.hidden = LIF_layer(size_in,
                                hidden_size,
                                forget,
                                spiker,
                                gradF,
                                gradS)
        self.out = LIF_layer(hidden_size,
                             size_out,
                             forget,
                             spiker,
                             gradF,
                             gradS)
        self.T = T
        self.size_out = size_out
        self.hidden_size = hidden_size

    def run_step(self, inpt, hist=False):
        # run one step of the simulation
        # for prediction, we do not need to store the history of hidden layers,
        # hence hist=False
        input = ber(inpt)
        self.out.run(self.hidden.run(input, hist), True)

    def run(self, inpt, hist=False):
        # run the all simulation
        for i in range(self.T):
            self.run_step(inpt, hist)

    def reset(self):
        # reset both layers
        self.hidden.reset()
        self.out.reset()

    def predict(self, inpt):
        # return the scaled sum of spikes for each class
        self.reset()
        self.run(inpt)
        return 1/self.T*np.sum(self.out.spikes, axis=0)

    def compute_grad(self, input, targets):
        # compute
        S = input.shape[0]
        hist_epsO = np.zeros((self.T, S, targets.shape[1]))
        hist_epsH = np.zeros((self.T, S, self.hidden.size_out))
        deltaO, epsO = None, None
        deltaH, epsH = None, None
        for t in range(self.T-1, -1, -1):
            deltaO, epsO = self.out.grad(last=(t == self.T-1),
                                         border=True,
                                         t=t,
                                         target=targets,
                                         W=None,
                                         deltaN=None,
                                         deltaT=deltaO,
                                         epsilonT=epsO,
                                         UN=None)
            deltaH, epsH = self.hidden.grad(last=(t == self.T-1),
                                            border=False,
                                            t=t,
                                            target=None,
                                            W=self.out.W,
                                            deltaN=deltaO,
                                            deltaT=deltaH,
                                            epsilonT=epsH,
                                            UN=self.out.potential[t])
            hist_epsO[t] = epsO
            hist_epsH[t] = epsH

        dbO = np.sum(hist_epsO, axis=(0, 1))
        dbH = np.sum(hist_epsH, axis=(0, 1))
        dWO = np.zeros_like(self.out.W)
        dWH = np.zeros_like(self.hidden.W)
        for t in range(self.T):
            dWO = dWO + np.dot(hist_epsO[t].T, self.hidden.spikes[t])
            dWH = dWH + np.dot(hist_epsH[t].T, input)
        return (dWH, dbH, dWO, dbO)


params = {
    "lr": .2,
    "beta1": .5,
    "beta2": .999,
    "eps": 1e-8
}


class Solver:
    def __init__(self,
                 batch_size=20,
                 n_epochs=10,
                 params=params,
                 update="sgd"):
        self.lr = params["lr"]
        self.b1 = params["beta1"]
        self.b2 = params["beta2"]
        self.eps = params["eps"]
        self.m = 0
        self.v = 0
        self.Th = 0
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.update = update
        self.mem = {}
        if self.update == "adam":
            for th in ["WH", "bH", "WO", "bO"]:
                self.mem[th] = {
                    "m": 0.5,
                    "v": 0.5
                }

    def adam_update(self, model, parameter, grad):
        mp = self.b1*self.mem[parameter]["m"] + (1-self.b1)*grad
        vp = self.b2*self.mem[parameter]["v"] + (1-self.b2)*grad*grad
        substract = self.lr*mp/(np.sqrt(vp) + self.eps)
        if parameter == "WH":
            model.hidden.W = model.hidden.W - substract
        elif parameter == "bH":
            model.hidden.bias = model.hidden.bias - substract
        elif parameter == "WO":
            model.out.W = model.out.W - substract
        else:
            model.out.bias = model.out.bias - substract
        self.mem[parameter]["m"] = mp
        self.mem[parameter]["v"] = vp

    def sgd_update(self, model, parameter, grad):
        if parameter == "WH":
            model.hidden.W = model.hidden.W - self.lr*grad
        elif parameter == "bH":
            model.hidden.bias = model.hidden.bias - self.lr*grad
        elif parameter == "WO":
            model.out.W = model.out.W - self.lr*grad
        else:
            model.out.bias = model.out.bias - self.lr*grad

    def step(self, model, input, target):
        model.reset()
        model.run(input, hist=True)
        (dWH, dbH, dWO, dbO) = model.compute_grad(input, target)
        if self.update == "sgd":
            self.sgd_update(model, "WH", dWH)
            self.sgd_update(model, "bH", dbH)
            self.sgd_update(model, "WO", dWO)
            self.sgd_update(model, "bO", dbO)
        else:
            self.adam_update(model, "WH", dWH)
            self.adam_update(model, "bH", dbH)
            self.adam_update(model, "WO", dWO)
            self.adam_update(model, "bO", dbO)
            print(model.out.bias)

    def train(self, model, x_train, y_train, x_test, y_test):
        n_samples = x_train.shape[0]
        mask = np.arange(n_samples)
        np.random.shuffle(mask)
        n_batch = n_samples // self.batch_size
        for n in range(self.n_epochs):
            loss = 0
            acc_train = 0
            for i in range(n_batch):
                idx = mask[i*self.batch_size: (i+1)*self.batch_size]
                x_batch = x_train[idx]
                y_batch = y_train[idx]
                self.step(model, x_batch, y_batch)
                logits = model.predict(x_batch)
                loss += 1/2/n_samples*np.sum(np.linalg.norm(y_batch - logits, axis=1))
                target = np.argmax(y_batch, axis=1)
                pred = np.argmax(logits, axis=1)
                acc_train += np.mean(pred == target)

            print("epoch", n+1, "has ended")
            if n % 2 == 0:
                acc_test = self.eval(model, x_test, y_test)
                print("Loss:", loss)
                print("Train Acc.:", acc_train/n_batch, "Test Acc.:", acc_test)
            self.lr = self.lr*0.85

    def eval(self, model, input, target):
        logits = model.predict(input)
        pred = np.argmax(logits, axis=1)
        targ = np.argmax(target, axis=1)
        return np.mean(pred == targ)
