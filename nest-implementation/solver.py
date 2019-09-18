import numpy as np

params = {
    "lr": .2
}


class Solver:
    def __init__(self, batch_size=20, n_epochs=10, lr=.2):
        self.lr = params["lr"]
        self.n_epochs = n_epochs
        self.batch_size = batch_size


    def sgd_update(self, model, parameter, grad):
        if parameter == "WH":
            model.hidden.W = model.hidden.W - self.lr*grad
        else:
            model.outpt.W = model.outpt.W - self.lr*grad


    def step(self, model, input, target):
        model.run(input)
        (dWH, dWO) = model.compute_grad(input, target)
        #self.sgd_update(model, "WH", dWH)
        model.updateW(dWH, hidden=True)
        #self.sgd_update(model, "WO", dWO)
        model.updateW(dWO, hidden=False)

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
