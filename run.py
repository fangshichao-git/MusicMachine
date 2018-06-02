'''
from midi import *
n,b=MidiAffairs.loadMidi(folder='C:\\Users\\fangs\\Desktop\\Temp\\a')
MidiAffairs.writeMidi(n[0],b[0],'C:\\Users\\fangs\\Desktop\\Temp\\a')
'C:\\Users\\fangs\\Desktop\\CS\\Project\\MusicMachine\\params'
'''
from mm import *
mm=MusicMachine()
#mm=MusicMachine('C:\\Users\\fangs\\Desktop\\CS\\Project\\MusicMachine\\params')
mm.train('C:\\Users\\fangs\\Desktop\\CS\\Project\\MusicMachine\\trainingSet_mini',10,180,display_sample=True)
mm.sing(100, verbose=False)