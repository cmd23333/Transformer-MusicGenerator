import numpy as np
import pretty_midi


def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''将琴声矩阵转化为.mid文件
    ----------
    piano_roll : np.ndarray, shape=(128,frames)
    program : The program number of the instrument.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm


def generate_from_random(unique_notes, seq_len=50):
    # 随机初始化
    generate = np.random.randint(0, unique_notes, seq_len).tolist()
    return generate


def generate_from_one_note(note_tokenizer, new_notes='35'):
    # 指定音符初始化
    generate = [note_tokenizer.notes_to_index['e'] for i in range(49)]
    generate += [note_tokenizer.notes_to_index[new_notes]]
    return generate


def generate_notes(generate, model, unique_notes, max_generated=1000, seq_len=50):
    # 生成序列
    for i in range(max_generated):
        test_input = np.array([generate])[:, i:i + seq_len]
        predicted_note = model.predict(test_input)
        predicted_note = predicted_note[:, -1, :]
        random_note_pred = np.random.choice(unique_notes + 1, 1, replace=False, p=predicted_note[0])
        generate.append(random_note_pred[0])
    return generate


def write_midi_file_from_generated(note_tokenizer, generate, midi_file_name="result.mid", start_index=49, fs=8, max_generated=1000):
    # 生成文件
    note_string = [note_tokenizer.index_to_notes[ind_note] for ind_note in generate]
    array_piano_roll = np.zeros((128, max_generated + 1), dtype=np.int16)
    for index, note in enumerate(note_string[start_index:]):
        if note == 'e':
            pass
        else:
            splitted_note = note.split(',')
            for j in splitted_note:
                array_piano_roll[int(j), index] = 1
    generate_to_midi = piano_roll_to_pretty_midi(array_piano_roll, fs=fs)
    print("Tempo {}".format(generate_to_midi.estimate_tempo()))
    for note in generate_to_midi.instruments[0].notes:
        note.velocity = 100
    generate_to_midi.write(midi_file_name)