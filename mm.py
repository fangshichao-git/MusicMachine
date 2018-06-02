
from model import *
from midi import MidiAffairs
from vecop import VecOp


# this is the entrance of the whole program.
class MusicMachine:

    # initialize the machine
    # details of the network isn't determined by the param, but determined by the code below. You may change the size/etc. by modifying the code.
    def __init__(self, trained_param=None):

        self.noteNet=simpleRNN(name='noteNet', inp_dim=88, hid_dim=60, out_dim=88, olayer_type='sigmoid')
        #self.beatNet=simpleRNN(name='beatNet', inp_dim=10, hid_dim=20, out_dim=10)

        if trained_param!=None: # we can load the pre-trained configurations! Yay!
            self.noteNet.loadParams()
            #self.beatNet.loadParams()



    def train(self, midi_path, en, eb, display_sample=False):
        noteData, beatData=MidiAffairs.loadMidi(midi_path)
        self.noteVec=VecOp.noteToVector(noteData, dim=self.noteNet.ninp)
        self.noteNet.train(data=self.noteVec, epoch=en) # learning rate not specified
        if display_sample:
            self.noteNet.displaySample(self.randomInitializerTrack(pool=self.noteVec))
        #self.beatNet.train(data=beatData, epoch=eb)
        self.noteNet.saveParams()
        #self.beatNet.saveParams()


    # yay! finally I am able to sing!
    def sing(self, length, verbose=False):
        #raw_outputs=self.noteNet.fantasize(round=length)
        raw_outputs=self.noteNet.fantasize(init_music=self.randomInitializerTrack(self.noteVec), round=length)
        for chord in raw_outputs:
            c2=[(i,chord[i]) for i in range(len(chord))]
            sc=sorted(c2, key=lambda x: x[1], reverse=True)
            if verbose:
                print('*****************************')
                for i in range(4):
                    print('Prob #%d, note=%d, prob=%lf\n'%(i,sc[i][0],sc[i][1]))
            
            
        noteVec=VecOp.binarify(raw_outputs)
        notes=VecOp.vecToNote(noteVec)
        MidiAffairs.writeMidi(notes,[240]*len(notes))


    def randomInitializerTrack(self, pool):
        return pool[np.random.randint(0,len(pool))]