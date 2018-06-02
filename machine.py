
from model import *
from midi import MidiAffairs
from vecop import VecOp


# this is the entrance of the whole program.
# a music machine can compose music.
# there are more literary names than this, but I can't think of any. :(
class MusicMachine:

    # initialize the machine by giving it a bunch of midi files and, optionally, trained parameter files.
    # details of the network are given in the code rather than in params, considering there are few occasions to alternate them.
    # you may change the size/etc. by modifying the code.
    def __init__(self, midi_path, trained_param=None):

        self.noteNet=simpleRNN(name='noteNet', inp_dim=88, hid_dim=60, out_dim=88, olayer_type='sigmoid', param_path=trained_param)
        self.beatNet=simpleRNN(name='beatNet', inp_dim=8, hid_dim=30, out_dim=8, olayer_type='softmax', param_path=trained_param)

        noteData, beatData=MidiAffairs.loadMidi(midi_path)
        self.noteVec=VecOp.noteToVector(noteData, dim=self.noteNet.ninp)
        self.beatVec=VecOp.beatToVector(beatData, dim=self.beatNet.ninp)


    # train the network.
    def train(self, en, eb):
        self.noteNet.train(data=self.noteVec, epoch=en) # learning rate not specified
        self.beatNet.train(data=self.beatVec, epoch=eb)


    # display a sample to show how well the network is trained.
    # for more details, go to models.py --> displaySample()
    def displaySample(self):
        n, b=self.randomInitializerTrack()
        self.noteNet.displaySample(n)
        #self.beatNet.displaySample(b)


    # save the trained parameters locally.
    # you can use the saved params as milestones in a very heavy training process.
    # or, aesthetically (make the mamchine your private composer), but I am sure that you prefer Mozart.
    def saveParams(self):
        self.noteNet.saveParams()
        self.beatNet.saveParams()


    # I am a bird. I can sing for any length if you feed me berries.
    # the "verbose" flag enables you to go deep into the bird's brain to see several largest values among the output layer across each step.
    # example: la=0.99, fa=0.8, do-sharp=0.1
    # keep in mind that it's a LOT OF information (and outputting this takes a long time).
    # the output file is saved as music.mid
    def sing(self, length, verbose=False):
        n, b=self.randomInitializerTrack()
        note_outputs=self.noteNet.fantasize(init_data=n, round=length, randomize_func=VecOp().binarify)
        for chord in note_outputs:
            c2=[(i,chord[i]) for i in range(len(chord))]
            sc=sorted(c2, key=lambda x: x[1], reverse=True)
            if verbose:
                print('*****************************')
                for i in range(4):
                    print('Prob #%d, note=%d, prob=%lf\n'%(i,sc[i][0],sc[i][1]))
        notes=VecOp.vecToNote(VecOp().binarify(note_outputs))
        beat_outputs=self.beatNet.fantasize(init_data=b, round=length)
        beats=VecOp.vecToBeat(beat_outputs)
        MidiAffairs.writeMidi(notes,beats)


    # returns the note sequence and beat sequence from a random midi file.
    # this is helpful for "burning-in" the network to a decent state.
    def randomInitializerTrack(self):
        return (self.noteVec[np.random.randint(0,len(self.noteVec))],
            self.beatVec[np.random.randint(0,len(self.beatVec))])