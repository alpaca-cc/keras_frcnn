from midiutil.MidiFile3 import MIDIFile
import numpy as np

note_defs = {
     17 : ("b6", 83),
     16 : ("a5", 81),
     15 : ("g5", 79),
     14 : ("f5", 77),
     13 : ("e5", 76),
     12 : ("d5", 74),
     11 : ("c5", 72),
      7 : ("b4", 71),
      6 : ("a4", 69),
      5 : ("g4", 67),
      4 : ("f4", 65),
      3 : ("e4", 64),
      2 : ("d4", 62),
      1 : ("c4", 60),
     27 : ("b3", 59),
     26 : ("a3", 57),
     25 : ("g3", 55),
     24 : ("f3", 53),
     23 : ("e3", 52),
     22 : ("d3", 50),
     21 : ("c3", 48),
      0 : ("rest", 0),
}

class Note(object):
    def __init__(self, note_key, duration):
        note_def = note_defs[note_key]
        self.note = note_def[0]
        self.pitch = note_def[1]
        self.duration = duration


# The format of the boxes would be [x1, y1, x2, y2, prob, cls_num]
def parse_boxes_to_notes(boxes, class_mapping):
    sorted_boxes = sorted(boxes, key=lambda v: v[0])
    notes = []
    for b in sorted_boxes:
        cls_num = int(b[-1])
        cls_name = class_mapping[cls_num]
        if cls_name == '-':
            #  double duration to the last note
            if len(notes) == 0:
                continue
            else:
                last_note = notes[-1]
                last_note.duration = duration * 2
        else:
            note = cls_name.split('_')
            note_key = int(note[0])
            # duration of a 4-tempo note would be 2
            duration = 8/float(note[1])
            notes.append(Note(note_key, duration))
    return notes


def generate_midi(notes):
    # Create the MIDIFile Object
    MyMIDI = MIDIFile(1)

    # Add track name and tempo. The first argument to addTrackName and
    # addTempo is the time to write the event.
    track = 0
    time = 0
    MyMIDI.addTrackName(track,time,"Sample Track")
    MyMIDI.addTempo(track,time, 120)

    # Add a note. addNote expects the following information:
    channel = 0
    # pitch = 60
    # duration = 1
    volume = 100

    # Now add the note.
    for note in notes:
        MyMIDI.addNote(track, channel, note.pitch, time, note.duration, volume)
        time += note.duration

    # And write it to disk.
    binfile = open("./server_files/output.mid", 'wb')
    MyMIDI.writeFile(binfile)
    binfile.close()


def run_midi_generate(boxes, class_mapping):
    notes = parse_boxes_to_notes(boxes, class_mapping)
    generate_midi(notes)


# 22_8:
# [  32.   85.   53.  128.] prob: 0.3766149878501892
# 3_8:
# [ 117.   85.  133.  133.] prob: 0.5155195593833923
# 4_16:
# [  58.   90.   80.  133.] prob: 0.9636281132698059
# 0_16:
# [  90.   85.  112.  138.] prob: 0.9995933175086975
# 14_8:
# [ 149.   80.  170.  144.] prob: 0.9982258677482605

# class_mapping = {0: '15_8', 1: '5_8', 2: '2_16', 3: '17_16', 4: '14_16', 5: '22_16', 6: '26_16', 7: '23_8', 8: '14_8', 9: '2_8', 10: '0_16', 11: '16_16', 12: '25_8', 13: '3_16', 14: '25_16', 15: '4_16', 16: '5_16', 17: '4_8', 18: '2_4', 19: '-', 20: '16_8', 21: '11_16', 22: '13_16', 23: '0_4', 24: '27_16', 25: '6_16', 26: '15_16', 27: '21_8', 28: '7_8', 29: '0_8', 30: '23_16', 31: '21_4', 32: '16_4', 33: '1_8', 34: '12_8', 35: '6_8', 36: '13_8', 37: '1_4', 38: '3_8', 39: '11_4', 40: '14_4', 41: '1_16', 42: '21_16', 43: '24_8', 44: '11_8', 45: '24_16', 46: '5_4', 47: '7_16', 48: '17_8', 49: '22_8', 50: '13_4', 51: '27_8', 52: '12_16', 53: '7_4', 54: '3_4', 55: '26_8', 56: '17_4', 57: '22_4', 58: '25_4', 59: '12_4', 60: '24_4', 61: '26_4', 62: '23_4', 63: '27_4', 64: '15_4', 65: '6_4', 66: '4_4', 67: 'bg'}
#
#
# sample_boxes = []
# sample_boxes.append(np.array([32., 85., 53., 128., 0.3766149878501892, 49.]))
# sample_boxes.append(np.array([117., 85., 133., 133., 5155195593833923, 38.]))
# sample_boxes.append(np.array([58., 90., 80., 133., 0.9636281132698059, 15.]))
# sample_boxes.append(np.array([90., 85., 112., 138., 0.9995933175086975, 10.]))
# sample_boxes.append(np.array([149., 80., 170., 144., 0.9982258677482605, 8.]))
#
# notes = parse_boxes_to_notes(sample_boxes, class_mapping)
# generate_midi(notes)