
import os
from mido import MidiFile, MidiTrack, Message
import numpy as np


# contains everything involving midi operation
class MidiAffairs:

    # load midis in the folder. Return an array of matrices.
    # heriachy: all->song->chord->note
    @staticmethod
    def loadMidi(folder):
        data=[]
        for file in list(filter(lambda x: x.endswith('.mid'), os.listdir(folder))):
            midi=MidiFile(os.path.join(folder,file))
            notes=[]
            for track in midi.tracks:
                abstime=0
                pending=[]
                for msg in list(filter(lambda x: x.type=='note_on' or x.type=='note_off', track)):
                    abstime+=msg.time
                    if msg.type=='note_on':
                        pending.append((msg.note, abstime))
                    else:
                        try:
                            on=next(filter(lambda x: x[0]==msg.note, pending))
                        except:
                            continue  # what a bad midi! Ignore this flawed note.
                        notes.append((msg.note, on[1], abstime-on[1])) # a tuple: (pitch, beginning_time, duration)
                        pending.remove(on)
            #notes.sort(key=lambda x: x[1]) # sort by beginning time
            notes=list(filter(lambda x: x[2]>0, sorted(notes, key=lambda x: x[1]))) # sort and filter blank notes

            timeline=[] # contains all time nodes for the separation in the next step.
            for note in notes:
                if not note[1] in timeline:
                    timeline.append(note[1])
                if not note[1]+note[2] in timeline:
                    timeline.append(note[1]+note[2])
            timeline.sort()

            chords=[]
            for i in range(len(timeline)-1):
                beg=timeline[i]
                end=timeline[i+1]
                dur=end-beg
                chord=[]
                splitted=[]
                while len(notes)>0:
                    note=notes[0]
                    if note[1]>=end:
                        break
                    if note[1]==beg and note[2]==dur:
                        chord.append(note)
                        notes.remove(note)
                    if note[1]==beg and note[2]>dur:
                        chord.append((note[0],note[1],dur))
                        notes.remove(note)
                        #notes.insert(0,(note[0],note[1]+dur,note[2]-dur))
                        splitted.append((note[0],note[1]+dur,note[2]-dur))
                notes=splitted+notes
                chords.append(chord)
            data.append(chords)
        
        # now separate data into two parts: note and beat
        noteData=[]
        beatData=[np.asarray([])]
        for song in data:
            nd=[]
            bd=[]
            for chord in song:
                onNotes=[]
                if len(chord)==0:
                    continue
                for note in chord:
                    onNotes.append(note[0])
                nd.append(onNotes)
                #nd.append([sorted(onNotes,reverse=True)[0]])
                bd.append(chord[0][2])
            noteData.append(nd)
            beatData.append(bd)

        return noteData, beatData


    # note: noteData and beatData in the params are actually different from the return values in the method above.
    # the return values of loadMidi is actually data of many midi files, so they're one dimension higher than those in writeMidi params.
    @staticmethod
    def writeMidi(noteData, beatData, outputDir='./'):
        midi=MidiFile()
        track=MidiTrack()
        midi.tracks.append(track)
        for i in range(len(noteData)):
            if len(noteData[i])==0:
                noteData[i].append(0) # to fill the blank with a "invisible" note.
            for note in noteData[i]:
                track.append(Message('note_on',note=note,velocity=63,time=0))
            track.append(Message('note_off',note=noteData[i][0],velocity=63,time=beatData[i]))
            for j in range(1,len(noteData[i])):
                track.append(Message('note_off',note=noteData[i][j],velocity=63,time=0))
        midi.save(filename=os.path.join(outputDir, 'music.mid'))