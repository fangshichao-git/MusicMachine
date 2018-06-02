'''
from midi import *
n,b=MidiAffairs.loadMidi(folder='C:\\Users\\fangs\\Desktop\\Temp\\a')
MidiAffairs.writeMidi(n[0],b[0],'C:\\Users\\fangs\\Desktop\\Temp\\a')
'C:\\Users\\fangs\\Desktop\\CS\\Project\\MusicMachine\\params'
'''
from machine import *
mm=MusicMachine()
#mm=MusicMachine('C:\\Users\\fangs\\Desktop\\CS\\Project\\MusicMachine\\params')
mm.train('C:\\Users\\fangs\\Desktop\\CS\\Project\\MusicMachine\\trainingSet',200,30)
mm.sing(200, verbose=False)