import torch
import torch.nn as nn

def get_collate_fn(pad_index):

  def collate_fn(batch):

    batch_en_ids = [example["en_ids"] for example in batch]
    batch_de_ids = [example["de_ids"] for example in batch]

    batch_en_ids = nn.utils.rnn.pad_sequence(sequences=batch_en_ids,padding_value=pad_index)
    batch_de_ids = nn.utils.rnn.pad_sequence(sequences=batch_de_ids,padding_value=pad_index)

    batch = {"en_ids":batch_en_ids,
             "de_ids":batch_de_ids}
    
    return batch

  return collate_fn

def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
  collate_fn = get_collate_fn(pad_index)
  data_loader = torch.utils.data.DataLoader(
      dataset=dataset,
      batch_size=batch_size,
      collate_fn=collate_fn,
      shuffle=shuffle
  )
  return data_loader