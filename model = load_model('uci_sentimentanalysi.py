model = load_model('uci_sentimentanalysis.h5')
with open( 'tokenizer.pickle', 'rb') as handle:
tokenizer = pickle.load(handle)