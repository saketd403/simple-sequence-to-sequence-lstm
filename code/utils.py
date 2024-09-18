import numpy as np
import torch
import torch.nn as nn
import argparse


def get_args():

  parser = argparse.ArgumentParser(description="Simple sequence to sequence mode using LSTM.")

  parser.add_argument('--batch_size',default=128,type=int,help='Batch size to be created by dataloaders.')
  parser.add_argument('--use_embedding',default=True,type=bool,help='Should you use pretrained embedding.')
  parser.add_argument('--epochs',default=30,type=int,help='Number of epochs for training the model.')
  parser.add_argument('--encoder_embedding_dim',default=300,type=int,help='Embedding dimension to be used by encoder.')
  parser.add_argument('--decoder_embedding_dim',default=300,type=int,help='Embedding dimension to be used by decoder.')
  parser.add_argument('--hidden_dim',default=512,type=int,help='Hidden dimension of LSTM.')
  parser.add_argument('--n_layers',default=2,type=int,help="Number of layers in stacked LSTM.")
  parser.add_argument('--encoder_dropout',default=0.5,type=float,help="Dropout for encoder model.")
  parser.add_argument('--decoder_dropout',default=0.5,type=float,help="Dropout for decoder model.")
  parser.add_argument('--max_length',default=1000,type=int,help="Max number of tokens in a sentence. The exceeding tokens will be ignored.")
  parser.add_argument('--lower',default=True,type=bool,help="Do u want to transform the text to lower case.")
  parser.add_argument('--min_freq',default=2,type=int,help="Only words with this much frequency will be considered to form vocabulary.")
  parser.add_argument('--clip',default=1.0,type=float,help="Clip the gradient norm by this much")
  parser.add_argument('--teacher_forcing_ratio',default=0.5,type=float,help="Teacher forcng ratio to be used by decoder.")
  parser.add_argument('--lr',default=0.001,type=float,help="Learning rate for training")

  parser.set_defaults(argument=True)

  return parser.parse_args()

def numericalize_example(example,en_vocab,de_vocab):

  en_ids = en_vocab.lookup_indices(example["en_tokens"])
  de_ids = de_vocab.lookup_indices(example["de_tokens"])

  return {"en_ids":en_ids,"de_ids":de_ids}

def create_embedding_matrix(word_to_vec, vocab, embedding_dim, unk_index):
    embedding_matrix = torch.randn(size=(len(vocab),embedding_dim))
    missed_word = []
    for word, idx in vocab.get_stoi().items():
        if word in word_to_vec:
            embedding_matrix[idx] = torch.tensor(word_to_vec[word],dtype=torch.float32)
        else:
          embedding_matrix[idx] = embedding_matrix[unk_index]
          missed_word.append(word)

    return missed_word,embedding_matrix

def load_glove_embeddings(file_path):
  word_to_vec = {}
  with open(file_path,"r",encoding="utf-8") as f:
    for line in f:
      values = line.split()
      word = values[0]
      vector = np.array(values[1:],dtype="float32")
      word_to_vec[word] = vector
  return word_to_vec


def tokenize_example(example,en_nlp,de_nlp,max_length,lower,sos_token,eos_token):

  en_tokens = [token.text for token in en_nlp.tokenizer(example["en"])][:max_length]
  de_tokens = [token.text for token in de_nlp.tokenizer(example["de"])][:max_length]

  if lower:
    en_tokens = [token.lower() for token in en_tokens]
    de_tokens = [token.lower() for token in de_tokens]

  en_tokens = [sos_token] + en_tokens + [eos_token]
  de_tokens = [sos_token] + de_tokens + [eos_token]

  return {"en_tokens":en_tokens,"de_tokens":de_tokens}

def init_weights(m):

  for name,parameter in m.named_parameters():
    nn.init.uniform_(parameter.data,a=-0.08,b=0.08)

def count_parameters(model):
  return sum(param.numel() for param in model.parameters() if param.requires_grad)

def translate_sentence(
    sentence,
    model,
    en_nlp,
    de_nlp,
    en_vocab,
    de_vocab,
    lower,
    sos_token,
    eos_token,
    device,
    max_output_length=25
):
  model.eval()
  with torch.no_grad():
    if isinstance(sentence, str):
      tokens = [token.text for token in de_nlp.tokenizer(sentence)]
    else:
      tokens = [token for token in sentence]
    
    if lower:
      tokens = [token.lower() for token in tokens]

    tokens = [sos_token] + tokens + [eos_token]

    ids = de_vocab.lookup_indices(tokens)

    tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)

    hidden, cell = model.encoder(tensor)

    inputs = en_vocab.lookup_indices([sos_token])

    for _ in range(max_output_length):
      inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
      output, hidden, cell = model.decoder(inputs_tensor,hidden, cell)
      predicted_token = output.argmax(-1).item()
      inputs.append(predicted_token)
      if predicted_token == en_vocab[eos_token]:
        break
    
    tokens = en_vocab.lookup_tokens(inputs)

  return tokens

