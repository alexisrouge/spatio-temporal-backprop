# Implementation of Wu et al Spatio-Temporal Back Propagation.
### Manual implementation of neurons

# Dynamic

* Differential equation of the neuron:
<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/567af50bd3fd349946df60784f3f8f04.svg?invert_in_darkmode" align=middle width=159.843255pt height=34.725404999999995pt/></p>
Then each time that the neuron's potential goes above the threshold <img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/9b63afbb51c19cd1b4284838eed439e3.svg?invert_in_darkmode" align=middle width=22.25091pt height=22.46574pt/>,
the neuron spikes and potential is reset to <img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/1414b15d7f385250542f02903cb9c796.svg?invert_in_darkmode" align=middle width=33.27489pt height=14.155350000000013pt/>

 This leads to the following numeric schema:


* input of neuron i at layer n at time t+1
<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/073e040ccafe3484e9f9147327246d38.svg?invert_in_darkmode" align=middle width=198.2145pt height=52.22613pt/></p>
which is:
<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/576ae19004f7d2b59589ae07b1f6d3fc.svg?invert_in_darkmode" align=middle width=161.50612499999997pt height=14.202787499999998pt/></p>

* voltage of neuron i of layer n at time t+1
<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/7997454288dfe4da2d07b31e0d426272.svg?invert_in_darkmode" align=middle width=240.62609999999998pt height=19.897019999999998pt/></p>
which is:
<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/9c761c1aeabb12fdeb129811c3be4c7b.svg?invert_in_darkmode" align=middle width=278.4705pt height=18.312359999999998pt/></p>

* output of neuron i of the layer n at time t+1
<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/60eed5ad3a804556dec3ca981e8b2371.svg?invert_in_darkmode" align=middle width=129.43524pt height=19.897019999999998pt/></p>
which is :
<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/958fa95abc666d50406ff9f9dc85a698.svg?invert_in_darkmode" align=middle width=138.068205pt height=18.312359999999998pt/></p>

with <img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/0edabc7d08813263b9cb07aeeb101411.svg?invert_in_darkmode" align=middle width=85.95972pt height=30.87678pt/> and <img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/3cea0b0bf51620d4aba4e613968783aa.svg?invert_in_darkmode" align=middle width=95.37891pt height=24.65759999999998pt/>

# Derivatives

Denote <img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/b12d24bc9bdbd5746e96c97b38f2b5de.svg?invert_in_darkmode" align=middle width=79.49204999999999pt height=29.954430000000002pt/> and <img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/870e099be781db90420f82b594cb9b0a.svg?invert_in_darkmode" align=middle width=80.51307pt height=29.954430000000002pt/>
with
<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/a397fcc3096c4fc495733183c2b8f490.svg?invert_in_darkmode" align=middle width=224.56334999999996pt height=47.60745pt/></p>

1. case <img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/cfa2de66160f66cc4d6de047181883c2.svg?invert_in_darkmode" align=middle width=39.743055000000005pt height=22.46574pt/> and <img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/8602049953802541e9cdb359b446a99e.svg?invert_in_darkmode" align=middle width=46.784595pt height=22.46574pt/>

<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/4fac51e7ec9ec68e5e0958496df62812.svg?invert_in_darkmode" align=middle width=217.93035pt height=48.182804999999995pt/></p>
and
<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/0afacb7f56be20f44c94e3ee0fd4f28a.svg?invert_in_darkmode" align=middle width=138.694875pt height=41.07245999999999pt/></p>

so in vector notations:

<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/e51e4f898a57e4c59896bc62f5198fc9.svg?invert_in_darkmode" align=middle width=225.92625pt height=47.60745pt/></p>
and
<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/62e1703923882db5e9d3f0c536fc0585.svg?invert_in_darkmode" align=middle width=169.0128pt height=33.812129999999996pt/></p>

2. case <img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/cfa2de66160f66cc4d6de047181883c2.svg?invert_in_darkmode" align=middle width=39.743055000000005pt height=22.46574pt/> and <img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/c205509eff098298d6240586ec39ccbd.svg?invert_in_darkmode" align=middle width=46.784595pt height=22.46574pt/>

<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/1bdfaffe2a5b76fa3c1200addb3f3699.svg?invert_in_darkmode" align=middle width=236.92844999999997pt height=52.22613pt/></p>
and
<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/6fc43bf34755feccec43ee99f58f2830.svg?invert_in_darkmode" align=middle width=128.134545pt height=41.07245999999999pt/></p>

so in vector notations:

<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/581519e3dae79fdad4c32cedc1446016.svg?invert_in_darkmode" align=middle width=270.8574pt height=33.812129999999996pt/></p>
and
<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/943f437e6792adf43d36008aa2f1b5de.svg?invert_in_darkmode" align=middle width=158.45247pt height=33.812129999999996pt/></p>


3. case <img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/adef3632c0a9e60a71ffb6cd8bab511a.svg?invert_in_darkmode" align=middle width=39.743055000000005pt height=22.46574pt/> and <img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/8602049953802541e9cdb359b446a99e.svg?invert_in_darkmode" align=middle width=46.784595pt height=22.46574pt/>

<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/70049a7057145d6fb44bdf0ede4c07f9.svg?invert_in_darkmode" align=middle width=296.65515pt height=41.07245999999999pt/></p>

and

<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/a440827dfcf6d9dd1c96d4f5fdf62b2e.svg?invert_in_darkmode" align=middle width=251.29995pt height=41.07245999999999pt/></p>

so in vector notations:
<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/04e361c281bad2d41a8182134ac2e9cf.svg?invert_in_darkmode" align=middle width=372.77955pt height=33.812129999999996pt/></p>
and
<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/ec25bad9a1d60dc0638548dee07c6afc.svg?invert_in_darkmode" align=middle width=307.58805pt height=33.812129999999996pt/></p>

4. case <img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/adef3632c0a9e60a71ffb6cd8bab511a.svg?invert_in_darkmode" align=middle width=39.743055000000005pt height=22.46574pt/> and <img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/15605c10841a9ee639ca78a40d052b1d.svg?invert_in_darkmode" align=middle width=46.784595pt height=22.46574pt/>
<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/54cfa9881c4e08fb3dba4cab49edbd6f.svg?invert_in_darkmode" align=middle width=421.01895pt height=52.22613pt/></p>
and
<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/792e8ba0e85c30b8cfde024bc6c8b785.svg?invert_in_darkmode" align=middle width=233.69939999999997pt height=40.28706pt/></p>

so in vector notations:

<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/ee90e05bcfeb2817b80e7c591b8c6dff.svg?invert_in_darkmode" align=middle width=522.126pt height=33.812129999999996pt/></p>

and

<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/9139784fbe113bd76679eb2bd4af8518.svg?invert_in_darkmode" align=middle width=289.9875pt height=33.812129999999996pt/></p>

Where <img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/449b35d6113d76b2c0b3cd32522d164e.svg?invert_in_darkmode" align=middle width=46.349160000000005pt height=22.46574pt/> is the componentwise product and <img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/bbe849e6f768c8d99d457a7b1132d981.svg?invert_in_darkmode" align=middle width=24.474945pt height=22.638659999999973pt/> is the transpose matrix of <img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/fb97d38bcc19230b0acd442e17db879c.svg?invert_in_darkmode" align=middle width=17.739810000000002pt height=22.46574pt/>

5. Derivative w.r.t <img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode" align=middle width=17.808285000000005pt height=22.46574pt/> and <img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode" align=middle width=7.054855500000005pt height=22.831379999999992pt/>

<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/b6b6b987851527fd61d8274d51c6c128.svg?invert_in_darkmode" align=middle width=198.91575pt height=47.60745pt/></p>
and
<p align="center"><img src="https://rawgit.com/alexisrouge/spatio-temporal-backprop/None/svgs/5d98032b42146225722c50987efb517a.svg?invert_in_darkmode" align=middle width=354.76649999999995pt height=47.60745pt/></p>
