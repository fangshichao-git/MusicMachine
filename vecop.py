
import random, math
import numpy as np


# contains methods about vector operations.
class VecOp:

    # convert original note data to pairs.
    @staticmethod
    def noteToVector(data, dim):
        vec=[]
        for music in data:
            v=[]
            for chord in music:
                c=np.zeros((1,dim))
                for note in chord:
                    c[0][note]=1
                    #c[0][note]=1/len(chord)
                v.append(c)
            vec.append(v)
        return np.asarray(vec)

    
    # binarify a matrix according to the assigned probabilities.
    # for example, a matrix before binarifying is [[0.4]], so it has 40 percent of possibility to become 1.
    @staticmethod
    def binarify(mat):
        res=[]
        for vec in mat:
            v2=[]
            for val in vec:
                #regulated=val*1.2-0.2
                regulated=val
                v2.append(int(random.random()<regulated))
                #v2.append(int(random.random()<((-math.log(1/val-1))/5+0.2))) # map the outputs of the sigmoid function to a probability range.
                #print('x=%lf, y=%lf, regulated=%lf'%((-math.log(1/val-1)),val,regulated))
            res.append(v2)
        return res


    @staticmethod
    def binarifyByGradThreshold(mat, max_chordsize=3, threshold=0.8, random_drop=False):
        res=[]
        for vec in mat:
            sv=sorted(vec,reverse=True)
            last=0
            while last<len(vec) and last<max_chordsize-1 and sv[last+1]>=sv[last]*threshold:
                last+=1
            onNotes=[np.where(vec==sv[i]) for i in range(last+1)]
            if random_drop:
                prob_keep=sv[:last+1]
                #print(prob_keep)
                for i,p in enumerate(prob_keep):
                    if random.random()>p/1.3:
                        #print('drop #%d'%i)
                        onNotes[i]=-1
                onNotes=list(filter(lambda x: x!=-1, onNotes))
            #print('onNotes='+str(onNotes))
            v2=np.zeros((len(vec)))
            for note in onNotes:
                v2[note]=1
            res.append(v2)
        return res


    @staticmethod
    def binarifyByAbsThreshold(mat, threshold=0.5):
        res=[]
        for vec in mat:
            res.append([int(x>threshold) for x in vec])
        return res


    @staticmethod
    def binarifyToSingleNote(mat, choice_range=3):
        res=[]
        for vec in mat:
            sv=sorted(vec,reverse=True)
            rnd=np.random.random_sample()*np.sum(sv[:choice_range])
            acc=0
            idx=0
            while(rnd>acc):
                acc+=sv[idx]
                idx+=1
            v2=np.zeros(len(vec))
            v2[np.where(vec==sv[idx-1])[0]]=1
            res.append(v2)
        return res


    @staticmethod
    def binarify3(mat, chordsize=3):
        res=[]
        for vec in mat:
            sv=sorted(vec,reverse=True)
            v2=np.zeros(len(vec))
            for value in sv[0:chordsize]:
                v2[np.where(vec==value)[0]]=1
            res.append(v2)
        return res
    

    # binarigy a matrix. assign 1 to a value if its original value is over threshold.
    @staticmethod
    def binarify4(mat, threshold=0.15):
        return (np.asarray(mat)>threshold).astype(int)

    
    # the reverse function of VecOp.noteToVector
    @staticmethod
    def vecToNote(mat):
        notes=[]
        for vec in mat:
            n=[]
            for i in range(len(vec)):
                if vec[i]==1:
                    n.append(i)
            notes.append(n)
        return notes
                    