import evaluate
import torch
from model import Seq2Seq, Encoder, Decoder
from engine import evaluate_fn
import numpy as np



def get_tokenizer_fn(nlp, lower):
    def tokenizer_fn(s):
        tokens = [token.text for token in nlp.tokenizer(s)]
        if lower:
            tokens = [token.lower() for token in tokens]
        return tokens

    return tokenizer_fn

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



def evaluate_model(model,test_data_loader,criterion,device,test_data,en_nlp,lower,translations):

    model.load_state_dict(torch.load("tut1-model.pt"))
    test_loss = evaluate_fn(model, test_data_loader, criterion, device)
    print(f"| Test loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):.3f}|")

    bleu = evaluate.load("bleu")

    predictions = [" ".join(translation[1:-1]) for translation in translations]

    references = [[example["en"]] for example in test_data]

    tokenizer_fn = get_tokenizer_fn(en_nlp, lower)

    results = bleu.compute(predictions=predictions, references=references, tokenizer=tokenizer_fn)

    return results