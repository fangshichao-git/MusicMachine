
import math
import os
import pickle

import numpy as np
import theano
import theano.tensor as T
from theano import function
from theano.tensor.nnet.nnet import sigmoid

from vecop import VecOp

# A model frame. Must be specialized with network types such as base_rnn or lstm

class Model:

    def __init__(self, inp_dim, out_dim):
        self.ninp=inp_dim
        self.nout=out_dim

    def train(self, data):
        pass

    def saveParams(self, path):
        pass

    # run the network from a random state for n rounds. returns the collection of their outputs.
    # this process is exactly the same when people goes into their fantastic worlds :)
    def fantasize(self, round):
        pass




# Instance 1: Simple RNN Network, using sigmoid activation function

class simpleRNN(Model):

    def __init__(self, name, inp_dim, hid_dim, out_dim, olayer_type):

        # initialize some values
        Model.__init__(self,inp_dim,out_dim)
        self.name=name
        self.nhid=hid_dim
        self.bound=20/math.sqrt(self.ninp)
        inp=T.tensor3()
        out=T.tensor3()
        hm1=T.tensor3()
        self.u=self.randomWeights(self.ninp,self.nhid)
        self.v=self.randomWeights(self.nhid,self.nout)
        self.w=self.randomWeights(self.nhid,self.nhid)

        # compile the forwardPass function. varies by the output layer type.
        # if you use chord as a unit, preferably choose sigmoid. if single notes are used, choose softmax.
        # because the network acts as a classifier.
        h=sigmoid(inp.dot(self.u)+hm1.dot(self.w))
        if olayer_type=='sigmoid':
            o=sigmoid(h.dot(self.v))
        else:
            #o=[T.nnet.softmax(h.dot(self.v))]
            o1=h.dot(self.v)
            (d1,d2,d3)=o1.shape
            o=T.nnet.softmax(o1.reshape((d1*d2,d3))).reshape((d1,d2,d3))
        self.forwardPass=theano.function(inputs=[inp,hm1], outputs=[o,h], allow_input_downcast=True)

        # compile the weight update function.
        if olayer_type=='sigmoid':
            loss=T.sum(T.pow(out-o,2))
        else:
            loss=T.sum(-out*T.log(o))
        self.calcLoss=theano.function(inputs=[o,out], outputs=loss, allow_input_downcast=True)
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


    def randomWeights(self, d1, d2):
        values=np.asarray(np.random.RandomState().uniform(
                    low=-np.sqrt(4* 6. / (d1 + d2)),
                    high=np.sqrt(4 * 6. / (d1 + d2)),
                    size=(d1, d2)),
                    dtype=theano.config.floatX)
        return theano.shared(value=values, borrow=True)


    def initHidden(self):
        return [(np.random.random(size=(1,self.nhid))-0.5)*self.bound]


    # initialize the hidden units by passing a part of a random human-composed track.
    def initHiddenByMusic(self, music, len_frac=1/3):
        hm1=self.initHidden()
        for i in range(int(len(music)*len_frac)):
            out, hm1=self.forwardPass([music[i]],hm1)
        return hm1


    # train the network. seu "pseudo-minibatch" learning (one batch=one midi file).
    # turns out to be better than online learning.
    # "data" is a collection of sequences of vectors (representing beat, note, etc.)
    # e.g. [[music1=[chord1],[chord2],...],[music2]]
    def train(self, data, epoch=500 ,lrate=0.011):
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


    def fantasize(self, init_data, round):
        # initialize the hm1 layer by going through part of a music in the training set.
        # this is the same as "letting the network finish the last part of a unfinished masterpiece".
        intro_frac=1/2
        hm1=self.initHiddenByMusic(init_data,len_frac=intro_frac)
        inp=[init_data[int(len(init_data)*intro_frac)+1]]
        fantasyDiary=[]
        for r in range(round):
            out, hm1=self.forwardPass(inp,hm1)
            fantasyDiary.append(out[0][0])
            inp=out
        return fantasyDiary


    # trace a learning process. preferably invoked after the net is trained.
    def displaySample(self, unit):
        # initialize the hidden units. this process is invisible (not displayed).
        hm1=self.initHiddenByMusic(music=unit, len_frac=1/3)
        # the display process now begins.
        for i in range(int(len(unit)/3),int(len(unit))-1):
            out, hm1=self.forwardPass([unit[i]],hm1)
            print('Desired output contains: '+str([j for j in range(len(unit[i+1][0])) if unit[i+1][0][j]!=0]))
            so=sorted(out[0][0],reverse=True)
            top5picks=[(np.where(value==out[0][0])[0][0],value) for value in so[:5]]
            print('The program predicts: '+str(top5picks)+'\n')


    # load the pre-trained parameters.
    def loadParams(self, path=''):
        path=os.path.join(path,'params')
        fu=open(os.path.join(path,'%s_u.params'%self.name),'rb')
        fv=open(os.path.join(path,'%s_v.params'%self.name),'rb')
        fw=open(os.path.join(path,'%s_w.params'%self.name),'rb')
        self.u=pickle.load(fu)
        self.v=pickle.load(fv)
        self.w=pickle.load(fw)


    # save the trained params into /params folder.
    def saveParams(self, path=''):
        path=os.path.join(path,'params')
        fu=open(os.path.join(path,'%s_u.params'%self.name),'wb')
        fv=open(os.path.join(path,'%s_v.params'%self.name),'wb')
        fw=open(os.path.join(path,'%s_w.params'%self.name),'wb')
        pickle.dump(self.u,fu)
        pickle.dump(self.v,fv)
        pickle.dump(self.w,fw)
