import numpy as np
import nest

params = {
    #"E_L": 0., # U_rest ie relaxed potential when I_e=0 (mV)
    "t_ref": .2, # refactory time (no spikes during this time) (ms)
    #"V_m": 0., # initial membrane potential (mV)
    #"V_min": -.5, # minimal potential that the membrane can reach (mV)
    "V_reset": -70., # potential just after firing (mV)
    "V_th": -55., # threshold potential (mV)
    #"C_m": 200., # capacity of the membrane (pF)
    "tau_m": 10. # neuron exp time constant (ms)
    #"I_e": 5. # input current (pA)
}

# R = tau_m/C_m
# max reachable potential with constant input current is V_reset + tau_m/C*I_e

def initW(size_in, size_out):
    # initialization function for network weights. Recommended form of Wu et al.
    W = np.random.random((size_out, size_in))
    sqrW = W**2
    normW = np.sqrt(np.sum(sqrW, axis=1)).reshape((size_out, 1))
    return W / normW

a = 1.

def gradS(x):
    # return -(x - V_th)/a*spiker(x)
    return 1/(np.sqrt(2*np.pi*a))*np.exp(-(x-params["V_th"])**2/(2*a))

def myDetector(x, V_th, reset):
    myTh = V_th*.95
    num_steps = x.shape[1]
    num_sample = x.shape[0]
    num_classes = x.shape[2]
    spikes = np.zeros_like(x)
    for i in range(num_sample):
        for k in range(num_classes):
            idx = np.arange(num_steps)[x[i,:,k] >= myTh]
            for j in idx:
                if j < num_steps - 1:
                    if np.abs(x[i, j+1, k] - reset) < .05:
                        spikes[i, j, k] = 1
                else:
                    spikes[i, j, k] = 1
    return spikes


class LIF_layer:
    def __init__(self, size_in, size_out, gradS=gradS, params=params, weight=1):
        self.W = initW(size_in, size_out)*weight
        self.neurons = nest.Create("iaf_psc_delta", size_out, params)
        #print(nest.GetStatus((self.neurons[0],)))
        self.multimeter = nest.Create("multimeter", params={"withtime":True,
                                    "record_from":["V_m"],
                                    "interval":.1})
        self.detector = nest.Create("spike_detector", params={"withgid": True,
                                                            "withtime": True,
                                                            "precise_times": True})
        nest.Connect(self.multimeter, self.neurons)
        nest.Connect(self.neurons, self.detector)
        self.size_out = size_out
        self.V_th = params["V_th"]
        self.reset = params["V_reset"]
        self.tau_m = params["tau_m"]
        self.step_size = .1
        self.gradS = gradS


    def reset(self):
        nest.ResetNetwork()

    def store(self, T, start):
        num_steps = int(T/self.step_size)
        num_classes = self.size_out
        num_sample = self.storePotential(T, num_steps, num_classes, start)
        self.storeSpikes(T, num_steps, num_classes, start, num_sample)

    def storePotential(self, T, num_steps, num_classes, start):
        report = nest.GetStatus(self.multimeter)[0]
        time_mask = report["events"]["times"] >= start
        num_sample = len(report["events"]["senders"][time_mask]) // (num_steps*num_classes)
        potential = np.zeros((num_sample, num_steps, num_classes))
        for i in range(num_classes):
            events = report["events"]["senders"][time_mask] == self.neurons[i] # idx of events of neuron[i]
            pot = report["events"]["V_m"][time_mask][events]
            potential[:,:, i] = pot.reshape((num_sample, num_steps))
        self.potential = potential
        # self.spikes = myDetector(potential, self.V_th, self.reset)
        return num_sample

    def storeSpikes(self, T, num_steps, num_classes, start, num_sample):
        report = nest.GetStatus(self.detector)[0]
        time_mask = report["events"]["times"] >= start
        spikes = np.zeros((num_sample, num_steps, num_classes))
        for i in range(num_classes):
            events = report["events"]["senders"][time_mask] == self.neurons[i]
            ts = (report["events"]["times"][time_mask][events] - start)*10
            ts = np.array(ts, int)
            times = np.zeros(num_sample*num_steps)
            times[ts] = 1
            spikes[:, :, i] = times.reshape((num_sample, num_steps))
        self.spikes = spikes


    def getSpikeCount(self, T, start, num_sample):
        num_classes = self.size_out
        num_steps = int(T/self.step_size)
        spike_count = np.zeros((num_sample, num_classes))
        report = nest.GetStatus(self.detector)[0]
        time_mask = report["events"]["times"] >= start
        for i in range(num_classes):
            events = report["events"]["senders"][time_mask] == self.neurons[i]  # spikes of neuron[i]
            ts = (report["events"]["times"][time_mask][events] - start)*10
            ts = np.array(ts, int)
            times = np.zeros(num_sample*num_steps)
            times[ts] = 1
            spike_count[:, i] = np.sum(times.reshape((num_sample, num_steps)), axis=1)
        return spike_count


    def grad(self, last, border, t, target, WN, deltaN, deltaT, epsilonT, UN):
        if last:
            if border:
                num_sample = self.spikes.shape[0] # number of samples
                num_steps = self.spikes.shape[1] # number of time step
                delta = - 1/num_steps/num_sample*(target - 1/num_steps*np.sum(self.spikes, axis=1))
            else:
                delta = np.dot(deltaN*self.gradS(UN), WN)
        else:
            delta = - self.V_th * deltaT * self.gradS(self.potential[:, t+1,:])
            if border:
                num_sample = self.spikes.shape[0] # number of samples
                num_steps = self.spikes.shape[1] # number of time step
                #print(delta.shape, target.shape, np.sum(self.spikes, axis=1).shape)
                delta = delta - 1/num_steps/num_sample*(target - 1/num_steps*np.sum(self.spikes, axis=1))
            else:
                delta = delta + np.dot(deltaN * self.gradS(UN), WN)

        epsilon = delta * self.gradS(self.potential[:, t,:])
        if not last:
            epsilon = epsilon + (1 - self.step_size/self.tau_m) * epsilonT
        return delta, epsilon


class LIF_network:
    def __init__(self,
                 size_in=784,
                 hidden_size=800,
                 size_out=10,
                 gradS=gradS,
                 T=30,
                 params=params):
        self.size_in = size_in
        self.hidden_size = hidden_size
        #self.inpt = nest.Create("step_current_generator", self.size_in)
        self.inpt = nest.Create("spike_generator", self.size_in)
        self.hidden = LIF_layer(size_in,
                                hidden_size,
                                gradS,
                                params,
                                weight=2.5)
        self.outpt = LIF_layer(hidden_size,
                             size_out,
                             gradS,
                             params,
                             weight=.75)
        nest.Connect(self.inpt, self.hidden.neurons, syn_spec={"weight": self.hidden.W, "delay": .1})
        nest.Connect(self.hidden.neurons, self.outpt.neurons, syn_spec={"weight": self.outpt.W, "delay": .1})
        self.i2h = nest.GetConnections(self.inpt, self.hidden.neurons)
        self.h2o = nest.GetConnections(self.hidden.neurons, self.outpt.neurons)
        self.T = T
        self.start = .1
        self.stop = .1


    def reset(self):
        nest.ResetNetwork()


    def updateW(self, dW, hidden=True):
        (size_out, size_in) = dW.shape
        if hidden:
            self.hidden.W = self.hidden.W - dW
            nest.SetStatus(self.i2h, [{"weight": self.hidden.W[i,j]} for i in range(size_out) for j in range(size_in)])
        else:
            self.outpt.W = self.outpt.W - dW
            nest.SetStatus(self.h2o, [{"weight": self.outpt.W[i,j]} for i in range(size_out) for j in range(size_in)])

    def run(self, inpt):
        # nest.ResetNetwork()
        self.spd = nest.Create("spike_detector", params={"withgid": True,
                                                "withtime": True,

                                                "precise_times": True})
        nest.Connect(self.inpt, self.spd)
        num_sample = inpt.shape[0]
        #print("num_sample:", num_sample)
        start = self.stop
        # print("start:", start)
        stop = start + self.T
        #print(t_step)
        for t in range(num_sample):
            indict = [{"start": start,
                       "stop": stop,
                       "spike_times": np.linspace(start, stop, inpt[t,i]*25),
                       "allow_offgrid_times":True
                       } for i in range(self.size_in)]
            nest.SetStatus(self.inpt, indict)
            nest.Simulate(self.T + .1*(start==.1)*(t==0))
            start = stop
            stop = start + self.T
        self.start = self.stop
        self.stop = num_sample*self.T + self.start


    def predict(self, inpt):
        num_sample = inpt.shape[0]
        self.run(inpt)
        S = self.outpt.getSpikeCount(self.T, self.start, num_sample)
        return 1./self.T*S


    def store(self):
        self.hidden.store(self.T, self.start)
        self.outpt.store(self.T, self.start)


    def compute_grad(self, input, targets):
        num_sample = input.shape[0]
        num_steps = int(self.T*10)
        self.store()
        hist_epsO = np.zeros((num_steps, num_sample, targets.shape[1]))
        hist_epsH = np.zeros((num_steps, num_sample, self.hidden_size))
        deltaO, epsO = None, None
        deltaH, epsH = None, None
        for t in range(num_steps-1, -1, -1):
            deltaO, epsO = self.outpt.grad(last=(t==num_steps-1),
                                        border=True,
                                        t=t,
                                        target=targets,
                                        WN=None,
                                        deltaN=None,
                                        deltaT=deltaO,
                                        epsilonT=epsO,
                                        UN=None)
            deltaH, epsH = self.hidden.grad(last=(t==num_steps-1),
                                            border=False,
                                            t=t,
                                            target=None,
                                            WN=self.outpt.W,
                                            deltaN=deltaO,
                                            deltaT=deltaH,
                                            epsilonT=epsH,
                                            UN=self.outpt.potential[:,t,:])
            hist_epsO[t] = epsO
            hist_epsH[t] = epsH

        dWO = np.zeros_like(self.outpt.W)
        dWH = np.zeros_like(self.hidden.W)
        for t in range(num_steps):
            dWO = dWO + np.dot(hist_epsO[t].T, self.hidden.spikes[:,t,:])
            dWH = dWH + np.dot(hist_epsH[t].T, input)
        return (dWH, dWO)
