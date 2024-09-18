import random
import numpy as np
import datasets
import torch.optim as optim
import spacy
from pathlib import Path
from torchtext.vocab import build_vocab_from_iterator
import tqdm

from utils import *
from data_setup import get_data_loader
from model import Seq2Seq, Encoder, Decoder
from engine import train
from evaluate_predictions import *

seed = 1234

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def get_datasets(dataset_name="bentrevett/multi30k"):

    dataset = datasets.load_dataset(dataset_name)

    print("The following is the description of dataset we are working with:")
    print(dataset)

    return (dataset["train"],dataset["validation"],dataset["test"])

def torch_transform(train_data,valid_data,test_data):
    data_type="torch"
    format_columns = ["en_ids","de_ids"]

    train_data = train_data.with_format(
        type=data_type,
        columns=format_columns,
        output_all_columns=True
    )

    valid_data = valid_data.with_format(
        type=data_type,
        columns=format_columns,
        output_all_columns=True

    )

    test_data = test_data.with_format(
        type=data_type,
        columns=format_columns,
        output_all_columns=True

    )

    return train_data, valid_data, test_data

def create_vocab(train_data,special_tokens,unk_token,pad_token,args):

    en_vocab = build_vocab_from_iterator(
            iterator = train_data["en_tokens"],
            min_freq=args.min_freq,
            specials=special_tokens
    )

    de_vocab = build_vocab_from_iterator(
            iterator = train_data["de_tokens"],
            min_freq=args.min_freq,
            specials=special_tokens
    )


    assert en_vocab[unk_token] == de_vocab[unk_token]
    assert en_vocab[pad_token] == de_vocab[pad_token]

    args.unk_index = en_vocab[unk_token]
    args.pad_index = en_vocab[pad_token]

    en_vocab.set_default_index(args.unk_index)
    de_vocab.set_default_index(args.unk_index)

    return en_vocab, de_vocab


def main():

    args = get_args()

    train_data, valid_data, test_data = get_datasets("bentrevett/multi30k")

    en_nlp = spacy.load("en_core_web_sm")
    de_nlp = spacy.load("de_core_news_sm")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sos_token = "<sos>"
    eos_token = "<eos>"
    unk_token="<unk>"
    pad_token="<pad>"

    special_tokens = [
        unk_token,
        pad_token,
        sos_token,
        eos_token
    ]

    tokenize_kwargs = {
        "en_nlp":en_nlp,
        "de_nlp":de_nlp,
        "max_length":args.max_length,
        "lower":args.lower,
        "sos_token":sos_token,
        "eos_token":eos_token
    }

    train_data = train_data.map(tokenize_example,fn_kwargs=tokenize_kwargs)
    valid_data = valid_data.map(tokenize_example,fn_kwargs=tokenize_kwargs)
    test_data = test_data.map(tokenize_example,fn_kwargs=tokenize_kwargs)

    en_vocab, de_vocab = create_vocab(train_data,special_tokens,unk_token,pad_token,args)


    if(args.use_embedding):

        #Get current directory
        current_directory = Path.cwd()

        # Setup path to data folder
        embedding_file_en = current_directory / "data" / "word_embeddings_en" / "glove.6B.300d.txt"
        embedding_file_de = current_directory / "data" / "word_embeddings_de" / "vectors.txt"

        word_to_vec_en = load_glove_embeddings(embedding_file_en)
        word_to_vec_de = load_glove_embeddings(embedding_file_de)

        missed_words_en,embedding_matrix_en = create_embedding_matrix(word_to_vec_en,en_vocab,args.decoder_embedding_dim,args.unk_index)
        missed_word_de,embedding_matrix_de = create_embedding_matrix(word_to_vec_de,de_vocab,args.encoder_embedding_dim,args.unk_index)
    else:

        embedding_matrix_en = torch.randn(size=(len(en_vocab),args.decoder_embedding_dim))
        embedding_matrix_de = torch.randn(size=(len(de_vocab),args.encoder_embedding_dim))

    numericalize_kwargs = {
    "en_vocab":en_vocab,
    "de_vocab":de_vocab
    }

    train_data = train_data.map(numericalize_example,fn_kwargs=numericalize_kwargs)
    valid_data = valid_data.map(numericalize_example,fn_kwargs=numericalize_kwargs)
    test_data = test_data.map(numericalize_example,fn_kwargs=numericalize_kwargs)

    train_data, valid_data, test_data = torch_transform(train_data,valid_data,test_data)


    train_data_loader = get_data_loader(train_data,args.batch_size,args.pad_index,shuffle=True)
    valid_data_loader = get_data_loader(valid_data,args.batch_size,args.pad_index)
    test_data_loader = get_data_loader(test_data,args.batch_size,args.pad_index)

    input_dim = len(de_vocab)
    output_dim = len(en_vocab)

    encoder = Encoder(
        input_dim,
        args.encoder_embedding_dim,
        args.hidden_dim,
        args.n_layers,
        args.encoder_dropout,
        args.pad_index,
        embedding_matrix_de
    )

    decoder = Decoder(
        output_dim,
        args.decoder_embedding_dim,
        args.hidden_dim,
        args.n_layers,
        args.decoder_dropout,
        args.pad_index,
        embedding_matrix_en
    )

    model = Seq2Seq(encoder,decoder,device).to(device)

    model.apply(init_weights)

    print(f"The model has {count_parameters(model):,} trainable parameters")

    optimizer = optim.Adam(model.parameters(),lr=args.lr)

    criterion = nn.CrossEntropyLoss(ignore_index=args.pad_index)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=0.1,
        min_lr=0.000001,
        patience=3,
        threshold=0.01,
        threshold_mode='abs'
    )


    #train(model,train_data_loader,valid_data_loader,optimizer,criterion,scheduler,device,args)

    translations = [
        translate_sentence(
            example["de"],
            model,
            en_nlp,
            de_nlp,
            en_vocab,
            de_vocab,
            args.lower,
            sos_token,
            eos_token,
            device,
        )
        for example in tqdm.tqdm(test_data)
    ]

    results = evaluate_model(model,test_data_loader,criterion,device,test_data,en_nlp,args.lower,translations)
    print("Evaluation results:")
    print(results)

if __name__ == "__main__":
    main()

