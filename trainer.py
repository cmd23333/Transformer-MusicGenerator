from util.data import *
import tensorflow as tf
from util.optimizer import CustomSchedule
from train import music_model, TrainModel
from util.play import generate_from_one_note, generate_notes, write_midi_file_from_generated


list_all_midi = get_list_midi()
note_tokenizer = NoteTokenizer()

for i in range(len(list_all_midi)):
    dict_time_notes = generate_dict_time_notes(list_all_midi, batch_song=1, start_index=i, fs=5)
    full_notes = process_notes_in_song(dict_time_notes)
    for note in full_notes:
        note_tokenizer.partial_fit(list(note.values()))

note_tokenizer.add_new_note('e')  # 添加空音符
unique_notes = note_tokenizer.unique_word  # 不同音符的个数，用于确定网络最后一层的输出维度


seq_len = 50
EPOCHS = 200
BATCH_SONG = 1
BATCH_NNET_SIZE = 64
TOTAL_SONGS = len(list_all_midi)
FRAME_PER_SECOND = 5
max_generate = 300


model = music_model(seq_len, unique_notes)
learning_rate = CustomSchedule(128)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss_fn = tf.keras.losses.sparse_categorical_crossentropy

train_class = TrainModel(EPOCHS, note_tokenizer, list_all_midi, FRAME_PER_SECOND,
                  BATCH_NNET_SIZE, BATCH_SONG, optimizer, loss_fn, TOTAL_SONGS, model, seq_len)
losses = train_class.train()


generate = generate_from_one_note(note_tokenizer, '69')
generate = generate_notes(generate, model, unique_notes, max_generate, seq_len)
write_midi_file_from_generated(generate, "one_note.mid", start_index=seq_len-1, fs=5, max_generated=max_generate)