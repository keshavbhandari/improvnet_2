# from improvnet.tokenizer.midi import MidiDict
# from improvnet.tokenizer.absolute import AbsTokenizer

# tokenizer = AbsTokenizer()
# # filepath = "/mnt/data/improvnet_data/Final_GigaMIDI_V1.1_Final/training-V1.1-80%/no-drums/54/82a5729dfa8f665548f325e90bac08f5.mid"
# # filepath = "/mnt/data/improvnet_data/doug_mcenzie_jazz/Time after Time 2.mid"
# # filepath = "/keshav/improvnet_2/improvnet/notebooks/beethoven_string-trio_3_6_(nc)wittenburg.mid"
# filepath = "/mnt/data/improvnet_data/Final_GigaMIDI_V1.1_Final/training-V1.1-80%/no-drums/24/58dfa7ff18afca6c6cf701d9e96d4432.mid"
# midi_dict = MidiDict.from_midi(filepath)
# tokens = tokenizer.tokenize(midi_dict)
# print(len(tokens))
# print(tokens)

# # Detokenize
# midi_dict_reconstructed = tokenizer.detokenize(tokens)
# midi_reconstructed = midi_dict_reconstructed.to_midi()
# midi_reconstructed.save("/keshav/improvnet_2/improvnet/notebooks/reconstructed.mid")

# # # Save midi_dict to midi
# # midi_reconstructed = midi_dict.to_midi()
# # midi_reconstructed.save("/keshav/improvnet_2/improvnet/notebooks/reconstructed.mid")

# Import mido and read file
import mido

filepath = "/mnt/data/improvnet_data/Final_GigaMIDI_V1.1_Final/training-V1.1-80%/no-drums/24/58dfa7ff18afca6c6cf701d9e96d4432.mid"
midi_original = mido.MidiFile(filepath)
print("Original MIDI:")
for i, track in enumerate(midi_original.tracks):
    print(f"Track {i}: {track.name}")
    for msg in track:
        print(msg)