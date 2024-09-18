
import tqdm
import torch.nn as nn
import torch 
import numpy as np

def train_fn(
    model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device
):

  model.train()
  epoch_loss = 0
  for i, batch in enumerate(data_loader):

    src = batch["de_ids"].to(device)
    trg = batch["en_ids"].to(device)

    optimizer.zero_grad()
    output = model(src,trg,teacher_forcing_ratio)
    output_dim = output.shape[-1]
    output = output[1:].view(-1,output_dim)

    trg = trg[1:].view(-1)

    loss = criterion(output,trg)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(),clip)
    optimizer.step()

    epoch_loss += loss.item()

  return epoch_loss/len(data_loader)


def evaluate_fn(
    model, data_loader, criterion, device
):

  model.eval()
  epoch_loss = 0
  with torch.no_grad():
    for i, batch in enumerate(data_loader):

      src = batch["de_ids"].to(device)
      trg = batch["en_ids"].to(device)

      output = model(src,trg,teacher_forcing_ratio=0)
      output_dim = output.shape[-1]
      output = output[1:].view(-1,output_dim)

      trg = trg[1:].view(-1)

      loss = criterion(output,trg)

      epoch_loss += loss.item()

  return epoch_loss/len(data_loader)


def train(model,train_data_loader,valid_data_loader,optimizer,criterion,scheduler,device,args):
    
    n_epochs = args.epochs
    clip = args.clip
    teacher_forcing_ratio = args.teacher_forcing_ratio

    best_valid_loss = float("inf")

    for epoch in tqdm.tqdm(range(n_epochs)):
        
        train_loss = train_fn(
            model,
            train_data_loader,
            optimizer,
            criterion,
            clip,
            teacher_forcing_ratio,
            device,
        )
      
        
        valid_loss = evaluate_fn(
            model,
            valid_data_loader,
            criterion,
            device,
        )
        scheduler.step(valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "tut1-model.pt")
        print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
        print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")
        