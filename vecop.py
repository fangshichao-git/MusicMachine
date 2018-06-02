
import random, math
import numpy as np


# contains methods about vector operations.
class VecOp:

    # convert original note data to pairs.
    @staticmethod
    def noteToVector(data, dim):
        lenLongestMusic=max([len(music) for music in data])
        m=np.zeros((len(data),lenLongestMusic,1,dim))
        for i in range(len(data)):
            for j in range(len(data[i])):
                for note in data[i][j]: # data[i][j] is a chord
                    m[i][j][0][note]=1
        return m


    @staticmethod
    def beatToVector(data, dim):
        vec=[]  
        lenLongestMusic=max([len(music) for music in data])
        m=np.zeros((len(data),lenLongestMusic,1,dim))
        for i in range(len(data)):
            longest=np.max(data[i]) # takes out the longest note
            standards=np.asarray([longest/math.pow(2,i) for i in range(dim-1)]+[0])
            for j in range(len(data[i])):
                diff=np.abs(standards-data[i][j])
                onNode=np.where(diff==np.min(diff))[0][0]
                m[i][j][0][onNode]=1
        return m
        '''
        for music in data:
            v=[]
            longest=np.max(music) # takes out the longest note
            standards=np.asarray([longest/math.pow(2,i) for i in range(dim-1)]+[0])
            for beat in music:
                b=np.zeros((1,dim))
                diff=np.abs(standards-beat)
                b[0][np.where(diff==np.min(diff))[0][0]]=1
                v.append(b)
            vec.append(v)
        return np.asarray(vec)
        '''

    
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
    def binarifyToSingle(mat, choice_range=2):
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
    

    # binarify a matrix. assign 1 to a value if its original value is over threshold.
    @staticmethod
    def binarify4(mat, threshold=0.15):
        return (np.asarray(mat)>threshold).astype(int)


    # binarify a matrix. select the largest value in a vector and mark all values larger than itself*threshold.
    @staticmethod
    def binarifyByComparitiveThreshold(mat, threshold=0.8):
        return (mat>np.max(mat)*0.8).astype(int)

    
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


    @staticmethod
    def vecToBeat(mat, maxdur=2048):
        beats=[]
        standards=[maxdur/math.pow(2,i) for i in range(len(mat[0]))]
        for vec in mat:
            beats.append(int(standards[np.where(vec==np.max(vec))[0][0]]))
        return beats