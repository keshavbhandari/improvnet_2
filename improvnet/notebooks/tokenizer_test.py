from improvnet.utils.utils import ProcessData

processor = ProcessData()

# Test the tokenizer with a sample MIDI file
midi_dict = processor.read_midi("/data/home/acw769/improvnet_2/improvnet/inference/2_marche.mid")
tokens = processor.midi_to_tokens(midi_dict)
print(f"Tokenized MIDI: {tokens[:50]} ...")  # Print first n tokens for brevity
print(f"Last tokens: {tokens[-50:]}")  # Print last n tokens for brevity

# Print vocabulary size
print(f"Vocabulary sizes: {processor.tokenizer.vocab_size}")
# Print tokens in vocab
print(f"First tokens in vocab: {list(processor.tokenizer.id_to_tok.items())[:100]}")
# Print id of special tokens
print(f"<MASK> token id: {processor.tokenizer.tok_to_id['<MASK>']}")
print(f"<BLANK> token id: {processor.tokenizer.tok_to_id['<BLANK>']}")
print(f"<P> token id: {processor.tokenizer.tok_to_id['<P>']}")
print(f"<SEP> token id: {processor.tokenizer.tok_to_id['<SEP>']}")

detokenized_midi_dict = processor.tokens_to_midi(tokens)
detokenized_midi_dict.save("/data/home/acw769/improvnet_2/improvnet/notebooks/reconstructed.mid")