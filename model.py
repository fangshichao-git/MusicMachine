
import math
import os
from six.moves import cPickle

import numpy as np
import theano
import theano.tensor as T
from theano import function
from theano.tensor.nnet.nnet import sigmoid

from vecop import VecOp



# a package of an entire simple recurrent neural network.
# feel free to create instances for any usage (even beyond music generation).
class simpleRNN:

    # initialize the network.
    # you can specify the output layer regularizing method by passing olayer_type='sigmoid' or 'softmax'.
    # loading the pre-trained network parameters is also supported.
    def __init__(self, name, inp_dim, hid_dim, out_dim, olayer_type, param_path=None):

        self.name=name
        self.ninp=inp_dim
        self.nhid=hid_dim
        self.nout=out_dim
        self.bound=20/math.sqrt(self.ninp) # a magic number for weight initialization.
        inp=T.matrix() # matrix for batch training.
        out=T.matrix()
        hm1=T.matrix()
        if param_path==None:
            self.u=self.randomWeights(self.ninp,self.nhid)
            self.v=self.randomWeights(self.nhid,self.nout)
            self.w=self.randomWeights(self.nhid,self.nhid)
        else:
            f=open(os.path.join(param_path,'params/')+self.name+'.param','rb')
            self.u=cPickle.load(f)
            self.v=cPickle.load(f)
            self.w=cPickle.load(f)
            f.close()

        # compile the forwardPass function. varies by the output layer type.
        # if you use chord as a unit, preferably choose sigmoid.
        # if single notes are used, choose softmax because the network acts as a classifier.
        h=sigmoid(inp.dot(self.u)+hm1.dot(self.w))
        if olayer_type=='sigmoid':
            o=sigmoid(h.dot(self.v))
        elif olayer_type=='softmax':
            o=T.nnet.softmax(h.dot(self.v))
        self.forwardPass=theano.function(inputs=[inp,hm1], outputs=[o,h], allow_input_downcast=True)

        # compile the loss function. also varies by olayer_type.
        if olayer_type=='sigmoid':
            loss=T.sum(T.pow(out-o,2))
        elif olayer_type=='softmax':
            loss=T.sum(-out*T.log(o))
        self.calcLoss=theano.function(inputs=[o,out], outputs=loss, allow_input_downcast=True)

        # compile the general weight update function.
        # the reason why it compiles this function even the networks uses pre-trained params is that\
        # I rarely use this program for aesthetic purposes. ;)
        du=T.grad(loss, self.u)
        dv=T.grad(loss, self.v)
        dw=T.grad(loss, self.w)
        learning_rate=T.scalar()
        self.calcGradients=theano.function(inputs=[inp,hm1,out], outputs=[du,dv,dw], allow_input_downcast=True)
        self.updateWeights=theano.function(inputs=[du,dv,dw,learning_rate], updates=[
            (self.u, self.u-learning_rate*du),
            (self.v, self.v-learning_rate*dv),
            (self.w, self.w-learning_rate*dw)
        ], allow_input_downcast=True)


    # this function is used for weight initialization.
    # the magic numbers are from a research paper.
    def randomWeights(self, d1, d2):
        values=np.asarray(np.random.RandomState().uniform(
                    low=-np.sqrt(4* 6. / (d1 + d2)),
                    high=np.sqrt(4 * 6. / (d1 + d2)),
                    size=(d1, d2)),
                    dtype=theano.config.floatX)
        return theano.shared(value=values, borrow=True)


    def initHidden(self):
        return [(np.random.random(size=(self.nhid))-0.5)*self.bound]


    # initialize the hidden units by passing a part of a random human-composed track.
    # when composing music, doing this instead of giving input and hidden layer random values will prevent\
    # the network from wierd behaviors.
    def initHiddenByMusic(self, music, len_frac=1/3):
        hm1=self.initHidden()
        for i in range(int(len(music)*len_frac)):
            out, hm1=self.forwardPass([music[i]],hm1)
        return hm1


    # train the network. uses full batch learning (considering the relatively small size of training set).
    # turns out to be better than online learning.
    # "data" is a batch of pieces of music, where
    # "music" is a collection of sequences of vectors (representing beat, note, etc.)
    # e.g. data=[[music1=[chord1],[chord2],...],[music2]]
    def train(self, data, epoch=500 ,lrate=0.15):
        for e in range(epoch):
            total_error=0
            total_iter=0
            hm1=np.asarray(self.initHidden()*len(data))
            for n in range(len(data[0])-1):
                du,dv,dw=self.calcGradients(data[:,n],hm1,data[:,n+1])
                out, hm1=self.forwardPass(data[:,n],hm1)
                self.updateWeights(du,dv,dw,lrate)
                total_error+=self.calcLoss(out,data[:,n+1])
                total_iter+=len(data)
            print('Epoch %d: Average Error = %lf\n'%(e,total_error/total_iter))        


    # run the network without disturbance for n rounds.
    # this process is exactly the same as that when people go into their fantastic worlds ;)
    # this function can be used to compose music.
    def fantasize(self, init_data, round, randomize_func=None):
        # initialize the hm1 layer by going through part of a music in the training set.
        # this is the same as "letting the network finish the last part of a unfinished masterpiece".
        intro_frac=1/2
        hm1=self.initHiddenByMusic(init_data,len_frac=intro_frac)
        inp=[init_data[int(len(init_data)*intro_frac)+1]]
        fantasyDiary=np.zeros((round,self.nout))
        for r in range(round):
            out, hm1=self.forwardPass(inp,hm1)
            fantasyDiary[r]=out[0]
            if randomize_func==None:
                inp=out
            else:
                inp=randomize_func(out)
        return fantasyDiary


    # display a sample in the training set to show how well the network is trained.
    # for example, a training set is [1,2,3,4,5]
    # then I input 1, and see if the network would output close to 2.
    # after updating the hidden layer, I'll input 2, and see whether it outputs 3.
    def displaySample(self, unit):
        # initialize the hidden units. this process is invisible (not displayed).
        hm1=self.initHiddenByMusic(music=unit, len_frac=1/3)
        # the display process now begins.
        for i in range(int(len(unit)/3),int(len(unit))-1):
            out, hm1=self.forwardPass([unit[i]],hm1)
            print('Desired output contains: '+str([j for j in range(len(unit[i+1])) if unit[i+1][j]!=0]))
            so=sorted(out[0],reverse=True)
            top5picks=[(np.where(value==out[0])[0][0],value) for value in so[:5]]
            print('The program predicts: '+str(top5picks)+'\n')


    # save the trained params into /params folder.
    def saveParams(self, path=''):
        f=open(os.path.join(path,'params/')+self.name+'.param','wb')
        for obj in [self.u, self.v, self.w]:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
