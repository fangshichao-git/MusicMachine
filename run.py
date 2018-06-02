'''
from midi import *
n,b=MidiAffairs.loadMidi(folder='C:\\Users\\fangs\\Desktop\\Temp\\a')
MidiAffairs.writeMidi(n[0],b[0],'C:\\Users\\fangs\\Desktop\\Temp\\a')
'C:\\Users\\fangs\\Desktop\\CS\\Project\\MusicMachine\\params'
'''
from machine import *
mm=MusicMachine(midi_path='C:\\Users\\fangs\\Desktop\\CS\\Project\\MusicMachine\\trainingSet_mini')
#mm=MusicMachine(midi_path='C:\\Users\\fangs\\Desktop\\CS\\Project\\MusicMachine\\trainingSet_mini', trained_param='./')
mm.train(80,0)
mm.displaySample()
mm.saveParams()
mm.sing(200, verbose=False)