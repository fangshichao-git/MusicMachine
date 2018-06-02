
from model import *
from midi import MidiAffairs
from vecop import VecOp


# this is the entrance of the whole program.
class MusicMachine:

    # initialize the machine
    # details of the network isn't determined by the param, but determined by the code below. You may change the size/etc. by modifying the code.
    def __init__(self, trained_param=None):

        self.noteNet=simpleRNN(name='noteNet', inp_dim=88, hid_dim=60, out_dim=88, olayer_type='sigmoid')
        self.beatNet=simpleRNN(name='beatNet', inp_dim=8, hid_dim=30, out_dim=8, olayer_type='softmax')

        if trained_param!=None: # we can load the pre-trained configurations! Yay!
            self.noteNet.loadParams()
            self.beatNet.loadParams()



    def train(self, midi_path, en, eb):
        noteData, beatData=MidiAffairs.loadMidi(midi_path)
        self.noteVec=VecOp.noteToVector(noteData, dim=self.noteNet.ninp)
        self.beatVec=VecOp.beatToVector(beatData, dim=self.beatNet.ninp)
        self.noteNet.train(data=self.noteVec, epoch=en) # learning rate not specified
        self.beatNet.train(data=self.beatVec, epoch=eb)
        self.noteNet.saveParams()
        self.beatNet.saveParams()


    def displaySample(self):
        n, b=self.randomInitializerTrack()
        self.noteNet.displaySample(n)
        self.beatNet.displaySample(b)


    # yay! finally I am able to sing!
    def sing(self, length, verbose=False):
        n, b=self.randomInitializerTrack()
        note_outputs=self.noteNet.fantasize(init_data=n, round=length)
        for chord in note_outputs:
            c2=[(i,chord[i]) for i in range(len(chord))]
            sc=sorted(c2, key=lambda x: x[1], reverse=True)
            if verbose:
                print('*****************************')
                for i in range(4):
                    print('Prob #%d, note=%d, prob=%lf\n'%(i,sc[i][0],sc[i][1]))
        notes_binarified=VecOp.binarifyByComparitiveThreshold(note_outputs)
        notes=VecOp.vecToNote(notes_binarified)
        beat_outputs=self.beatNet.fantasize(init_data=b, round=length)
        beats=VecOp.vecToBeat(beat_outputs)
        MidiAffairs.writeMidi(notes,beats)


    def randomInitializerTrack(self):
        return (self.noteVec[np.random.randint(0,len(self.noteVec))],
            self.beatVec[np.random.randint(0,len(self.beatVec))])