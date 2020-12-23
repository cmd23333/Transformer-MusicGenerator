import glob
import random
import pretty_midi
import numpy as np


def get_list_midi(folder='data/*.mid'):
    all_midi = glob.glob(folder)
    random.shuffle(all_midi)
    return all_midi


class NoteTokenizer:
    def __init__(self):
        self.notes_to_index = {}
        self.index_to_notes = {}
        self.num_of_word = 0
        self.unique_word = 0
        self.notes_freq = {}

    def transform(self, list_array):
        """ 把一段音符（字符串）转化为一段音符（索引）
        如论文4.4所述
        """
        transformed_list = []
        for instance in list_array:
            transformed_list.append([self.notes_to_index[note] for note in instance])
        return np.array(transformed_list, dtype=np.int32)

    def partial_fit(self, notes):
        """ （部分）自动匹配索引到音符，音符到索引的字典，除了空音符'e'
        """
        for note in notes:
            note_str = ','.join(str(a) for a in note)
            if note_str in self.notes_freq:
                self.notes_freq[note_str] += 1
                self.num_of_word += 1
            else:
                self.notes_freq[note_str] = 1
                self.unique_word += 1
                self.num_of_word += 1
                self.notes_to_index[note_str], self.index_to_notes[self.unique_word] = self.unique_word, note_str

    def add_new_note(self, note):
        """ 主动为索引字典添加新音符，用于添加空音符'e'
        """
        assert note not in self.notes_to_index
        self.unique_word += 1
        self.notes_to_index[note], self.index_to_notes[self.unique_word] = self.unique_word, note


def generate_batch_song(list_all_midi, batch_music=16, start_index=0, fs=30, seq_len=50):
    """
    生成送入神经网络的输入和输出
    详见论文4.3

    Returns
    Tuple of input and target neural network
    """
    assert len(list_all_midi) >= batch_music
    dict_time_notes = generate_dict_time_notes(list_all_midi, batch_music, start_index, fs)

    list_musics = process_notes_in_song(dict_time_notes, seq_len)
    collected_list_input, collected_list_target = [], []

    for music in list_musics:
        list_training, list_target = generate_input_and_target(music, seq_len)
        collected_list_input += list_training
        collected_list_target += list_target
    return collected_list_input, collected_list_target


def generate_dict_time_notes(list_all_midi, batch_song=16, start_index=0, fs=30):
    """.mid音乐文件=》琴声矩阵
    详见论文4.1

    Returns
    dictionary of music to piano_roll (in np.array)
    """
    assert len(list_all_midi) >= batch_song

    dict_time_notes = {}
    process_midi = range(start_index, min(start_index + batch_song, len(list_all_midi)))
    for i in process_midi:
        midi_file_name = list_all_midi[i]
        try:  # Handle exception on malformat MIDI files
            midi_pretty_format = pretty_midi.PrettyMIDI(midi_file_name)
            piano_midi = midi_pretty_format.instruments[0]  # Get the piano channels
            piano_roll = piano_midi.get_piano_roll(fs=fs)
            dict_time_notes[i] = piano_roll
        except Exception as e:
            print(e)
            print("broken file : {}".format(midi_file_name))
            pass
    return dict_time_notes


def generate_input_and_target(dict_keys_time, seq_len=50):
    """ 生成适合模型的输入(Inputs)和输出(Target)
    详见论文4.3

    Parameters
    ==========
    dict_keys_time : Dictionary of timestep and notes
    """
    # Get the start time and end time
    start_time, end_time = list(dict_keys_time.keys())[0], list(dict_keys_time.keys())[-1]
    list_training, list_target = [], []
    for index_enum, time in enumerate(range(start_time, end_time)):
        list_append_training, list_append_target = [], []
        start_iterate = 0

        if index_enum < seq_len:
            start_iterate = seq_len - index_enum - 1
            for i in range(start_iterate):  # add 'e' to the seq list.
                list_append_training.append('e')

        for i in range(start_iterate, seq_len):
            index_enum = time - (seq_len - i - 1)
            if index_enum in dict_keys_time:
                list_append_training.append(','.join(str(x) for x in dict_keys_time[index_enum]))
            else:
                list_append_training.append('e')

        # add time + 1 to the list_append_target
        if time + 1 in dict_keys_time:
            list_append_target.append(','.join(str(x) for x in dict_keys_time[time + 1]))
        else:
            list_append_target.append('e')
        list_training.append(list_append_training)
        list_target.append(list_append_target)
    return list_training, list_target


def process_notes_in_song(dict_time_notes, seq_len=50):
    """
    将琴声矩阵转化为字典
    详见论文4.2

    Parameters
    ==========
    dict_time_notes :dict contains index of music ( in index ) to piano_roll (in np.array)
    """
    list_of_dict_keys_time = []

    for key in dict_time_notes:
        sample = dict_time_notes[key]
        times = np.unique(np.where(sample > 0)[1])  # 排个序其实
        index = np.where(sample > 0)
        dict_keys_time = {}

        for time in times:
            index_where = np.where(index[1] == time)
            notes = index[0][index_where]
            dict_keys_time[time] = notes
        list_of_dict_keys_time.append(dict_keys_time)
    return list_of_dict_keys_time


