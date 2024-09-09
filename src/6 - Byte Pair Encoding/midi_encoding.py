# NOTE - This is almost the same code as from the previous notebooks, however it doesn't
# pad the idx score SOS up to block_size as it is for a transformer model and so can handle
# variable length sequences. Also increased the resolution to 32nd notes.

import numpy as np
from itertools import chain
from itertools import groupby
from functools import reduce
from typing import Collection, List
from pathlib import Path
import music21 as m21
musescore_path = '/usr/bin/mscore'
m21.environment.set('musicxmlPath', musescore_path)
m21.environment.set('musescoreDirectPNGPath', musescore_path)

BEATS_PER_MEASURE = 4
SAMPLES_PER_BEAT = 8 # i.e. 4 beats per bar and 8 samples per beat gives a resolution of 32nd notes
SAMPLES_PER_BAR = BEATS_PER_MEASURE * SAMPLES_PER_BEAT
MIDI_NOTE_COUNT = 128
MAX_NOTE_DUR = 8 * SAMPLES_PER_BAR # 8 bars of 32nd notes
SEPARATOR_IDX = -1 # separator value for numpy encoding
MIDI_NOTE_RANGE = (0, 127)
SOS = '<|sos|>' # Start of sequence
EOS = '<|eos|>' # End of sequence
SEP = '<|sep|>' # End of timestep (required for polyphony). Note index -1
PAD = '<|pad|>' # Padding to ensure blocks are the same size
 # SEP token must be last, i.e. one place before note tokens, so that adding the note offset still works when encoding
SPECIAL_TOKENS = [SOS, EOS, PAD, SEP]
NOTE_TOKENS = [f'n{i}' for i in range(MIDI_NOTE_COUNT)]
DURATION_SIZE = MAX_NOTE_DUR + 1 # + 1 for 0 length
DURATION_TOKENS = [f'd{i}' for i in range(DURATION_SIZE)]
NOTE_START, NOTE_END = NOTE_TOKENS[0], NOTE_TOKENS[-1] 
DURATION_START, DURATION_END = DURATION_TOKENS[0], DURATION_TOKENS[-1]
ALL_TOKENS = SPECIAL_TOKENS + NOTE_TOKENS + DURATION_TOKENS
TIMESIG = f'{BEATS_PER_MEASURE}/4'

def stream_to_sparse_enc(stream_score, note_size=MIDI_NOTE_COUNT, sample_freq=SAMPLES_PER_BEAT, max_note_dur=MAX_NOTE_DUR):    
    # Time is measured in quarter notes since the start of the piece

    # (MusicAutobot author:) TODO: need to order by instruments most played and filter out percussion or include the channel
    highest_time = max(
        stream_score.flatten().getElementsByClass('Note').stream().highestTime,
        stream_score.flatten().getElementsByClass('Chord').stream().highestTime)
    
    # Calculate the maximum number of time steps
    max_timestep = round(highest_time * sample_freq) + 1
    sparse_score = np.zeros((max_timestep, len(stream_score.parts), note_size), dtype=np.int32)

    # Convert a note to a tuple of (pitch,offset,duration)
    def note_data(pitch, note):
        return (pitch.midi, int(round(note.offset*sample_freq)), int(round(note.duration.quarterLength*sample_freq)))

    for idx, part in enumerate(stream_score.parts):
        
        notes = chain.from_iterable(
            [note_data(elem.pitch, elem)] if isinstance(elem, m21.note.Note)
            else [note_data(p, elem) for p in elem.pitches] if isinstance(elem, m21.chord.Chord) 
            else []
            for elem in part.flatten()
        )

        # sort flattened note list by timestep (1), duration (2) so that hits are not overwritten and longer notes have priority
        notes_sorted = sorted(notes, key=lambda x: (x[1], x[2])) 

        for note in notes_sorted:
            if note is not None:
                pitch, timestep, duration = note
                clamped_duration = max_note_dur if max_note_dur is not None and duration > max_note_dur else duration
                sparse_score[timestep, idx, pitch] = clamped_duration
    
    return sparse_score

# Pass in the 'one-hot' encoded numpy score
def sparse_to_position_enc(sparse_score, skip_last_rest=True):

    def encode_timestep(acc, timestep):
        encoded_timesteps, wait_count, tidx = acc
        encoded_timestep = timestep_to_position_enc(timestep, tidx) # pass in all notes for both instruments, merged list returned
        if len(encoded_timestep) == 0: # i.e. all zeroes at time step
            wait_count += 1
        else:
            if wait_count > 0:
                separator_position = tidx - wait_count
                encoded_timesteps.append([SEPARATOR_IDX, wait_count, separator_position]) # add rests
            encoded_timesteps.extend(encoded_timestep)
            wait_count = 1
        
        return encoded_timesteps, wait_count, tidx + 1
    
    # encoded_timesteps is an array of [ pitch, duration, position ]
    encoded_timesteps, final_wait_count, final_tidx = reduce(encode_timestep, sparse_score, ([], 0, 0))

    if final_wait_count > 0 and not skip_last_rest:
        encoded_timesteps.append([SEPARATOR_IDX, final_wait_count, final_tidx]) # add trailing rests

    return np.array(encoded_timesteps).reshape(-1, 3) # reshaping. Just in case result is empty
    
def timestep_to_position_enc(timestep, tidx, note_range=MIDI_NOTE_RANGE):

    note_min, note_max = note_range
    position = tidx

    def encode_note_data(note_data, active_note_idx):
        instrument, pitch = active_note_idx
        duration = timestep[instrument, pitch]
        if pitch >= note_min and pitch < note_max:
            note_data.append([pitch, duration, position, instrument])
        return note_data
    
    active_note_idxs = zip(*timestep.nonzero())
    encoded_notes = reduce(encode_note_data, active_note_idxs, [])
    sorted_notes = sorted(encoded_notes, key=lambda x: x[0], reverse=True) # sort by note (highest to lowest)

    # Dropping instrument information for simplicity.
    # MusicAutobot allows different encoding schemes which include instrument number and split pitch into class / octave.
    return [n[:3] for n in sorted_notes]

def position_to_idx_enc(note_position_score, vocab):
    note_idx_score, pos_score = vocab.encode(note_position_score)

    prefix =  np.array([vocab.sos_idx])
    prefix_position = np.array([pos_score[0]])

    suffix = np.array([vocab.eos_idx])
    suffix_position = np.array([pos_score[-1]])

    note_idx_score = np.concatenate([prefix, note_idx_score.reshape(-1), suffix])
    pos_score = np.concatenate([prefix_position, pos_score, suffix_position])

    # Returning note and positions in stacked array as we want to embed them separately in the model
    return np.stack([note_idx_score, pos_score], axis=1)

def import_midi_file(file_path):
    midifile = m21.midi.MidiFile()
    if isinstance(file_path, bytes):
        midifile.readstr(file_path)
    else:
        midifile.open(file_path)
        midifile.read()
        midifile.close()
    return midifile

def midifile_to_stream(midifile): 
    return m21.midi.translate.midiFileToStream(midifile)

def midifile_to_idx_score(file_path, vocab):
    midifile = import_midi_file(file_path)
    stream = midifile_to_stream(midifile)
    if stream.getTimeSignatures()[0].ratioString == '4/4':
        sparse_score = stream_to_sparse_enc(stream)
        note_pos_score = sparse_to_position_enc(sparse_score)
        return position_to_idx_enc(note_pos_score, vocab)
    else:
        return None

# Combining notes with different durations into a single chord may overwrite conflicting durations.
def group_notes_by_duration(notes):
    get_note_quarter_length = lambda note: note.duration.quarterLength
    sorted_notes = sorted(notes, key=get_note_quarter_length)
    return [list(g) for k,g in groupby(sorted_notes, get_note_quarter_length)]

def sparse_instrument_to_stream_part(sparse_instrument_score, step_duration):
    part = m21.stream.Part()
    part.append(m21.instrument.Piano())
    
    for t_idx, pitch_values in enumerate(sparse_instrument_score):

        def decode_sparse_note(notes, pitch_index):
            note = m21.note.Note(pitch_index)
            quarters = sparse_instrument_score[t_idx, pitch_index]
            note.duration = m21.duration.Duration(float(quarters) * step_duration.quarterLength)
            notes.append(note)
            return notes
        
        pitch_idxs = np.nonzero(pitch_values)[0]

        if len(pitch_idxs) != 0: 
            notes = reduce(decode_sparse_note, pitch_idxs, [])
            for note_group in group_notes_by_duration(notes):
                note_position = t_idx*step_duration.quarterLength
                if len(note_group) == 1:
                    part.insert(note_position, note_group[0])
                else:
                    chord = m21.chord.Chord(note_group)
                    part.insert(note_position, chord)

    return part

def sparse_to_stream_enc(sparse_score, bpm=120):
    step_duration = m21.duration.Duration(1. / SAMPLES_PER_BEAT)
    stream = m21.stream.Score()
    stream.append(m21.meter.TimeSignature(TIMESIG))
    stream.append(m21.tempo.MetronomeMark(number=bpm))

    # Not required here but left as example of options available
    stream.append(m21.key.KeySignature(0))
    
    for inst in range(sparse_score.shape[1]):
        part = sparse_instrument_to_stream_part(sparse_score[:,inst,:], step_duration)
        stream.append(part)
    
    # Again, not required yet but left as example
    stream = stream.transpose(0)
    
    return stream

def position_to_sparse_enc(note_position_score):

    # Add all the separator durations as they denote the elapsed time
    score_length = sum(timestep[1] for timestep in note_position_score if timestep[0] == SEPARATOR_IDX) + 1
    
    # Single instrument as we discarded the instrument information when encoding
    # We will adapt to handle multiple instruments later
    instrument = 0

    def decode_note_position_step(acc, note_pos_step):
        timestep, sparse_score = acc
        note, duration = note_pos_step.tolist()
        if note < SEPARATOR_IDX:  # Skip special token
            return acc
        elif note == SEPARATOR_IDX:  # Time elapsed
            return (timestep + duration, sparse_score)
        else:
            sparse_score[timestep, instrument, note] = duration
            return (timestep, sparse_score)

     # (timesteps, instruments, pitches)
    initial_sparse_score = np.zeros((score_length, 1, MIDI_NOTE_COUNT))
    _, final_sparse_score = reduce(decode_note_position_step, note_position_score, (0, initial_sparse_score))

    return final_sparse_score

# No validation of note position encoding included to keep it simple for now
def idx_to_position_enc(idx_score, vocab):
    # Filter out special tokens, convert idxs to positions and pair up note / durations
    notes_durs_start, notes_durs_end = vocab.note_position_enc_range # range of non-special token values
    notes_durations_idx_score = idx_score[np.where((idx_score >= notes_durs_start) & (idx_score < notes_durs_end))]
    return vocab.decode(notes_durations_idx_score).reshape(-1, 2)

def idx_to_stream_enc(idx_score, vocab):
    position_score = idx_to_position_enc(idx_score, vocab)
    sparse_score = position_to_sparse_enc(position_score)
    return sparse_to_stream_enc(sparse_score)