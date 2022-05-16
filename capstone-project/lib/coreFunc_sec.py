import faiss
import json
import torch
from math import log10
from time import time
import numpy as np
from torch.nn import TripletMarginLoss,CrossEntropyLoss,MSELoss

def my_Backpropagation(loss,My_model_optimizer,train_loss,current,batch,size,length):

    My_model_optimizer.zero_grad()
    loss.backward()
    My_model_optimizer.step()
    loss = loss.item()
    train_loss += loss
    current += length

    if batch % 5 == 0:
        print(f"loss: {loss:7f}  [{current:5d}/{size:5d}]", end='\r')

    return train_loss,current

def train_loop(PBert_trainloader, NBert_trainloader, Bert_trainloader, Resnet_trainloader, My_model,  My_model_optimizer,
                Category_trainloader, step, device):
    '''
    train a model in an epoch
    '''
    size = len(Resnet_trainloader.dataset)
    current = 0
    train_loss = 0
    My_model.train()
    batch = 0
    classifier_loss = CrossEntropyLoss()
    triplet_loss = TripletMarginLoss(margin=0.2, p=2)
    
    if step == 1:
        for batch, ((R,PR,NR), C) in enumerate(zip(Resnet_trainloader, Category_trainloader)):
            R,PR,NR= R.float(),PR.float(),NR.float()
            R,PR,NR,C = R.to(device), PR.to(device), NR.to(device), C.to(device)
            
            R_pred = My_model.ResNet(R)
            PR_pred = My_model.ResNet(PR)
            NR_pred = My_model.ResNet(NR)
            loss = triplet_loss(R_pred,PR_pred,NR_pred)
            train_loss, current = my_Backpropagation(loss,My_model_optimizer,train_loss,current,batch,size,len(R))

    elif step == 2:
        for batch, ((R,PR,NR), C) in enumerate(zip(Resnet_trainloader, Category_trainloader)):
            R,PR,NR= R.float(),PR.float(),NR.float()
            R,PR,NR,C = R.to(device), PR.to(device), NR.to(device), C.to(device)
            
            R_pred = My_model.ResNet(R)
            PR_pred = My_model.ResNet(PR)
            NR_pred = My_model.ResNet(NR)
            C_pred = My_model.Classification(R_pred)
            loss = classifier_loss(C_pred,C) + triplet_loss(R_pred,PR_pred,NR_pred)
            train_loss, current = my_Backpropagation(loss,My_model_optimizer,train_loss,current,batch,size,len(R))

    print()
    train_loss /= (batch + 1)
    print(f"Train Error: \n Avg loss: {train_loss:8f} \n")

    return train_loss


def test_loop(Bert_valloader, Resnet_valloader, My_model, Category_valloader, device, mode):
    '''
    evaluate model in validation stage or test stage
    '''
    size = len(Resnet_valloader.dataset)
    current = 0
    test_loss = 0
    batch = 0
    My_model.eval()
    classifier_loss = CrossEntropyLoss()
    mse_loss = MSELoss(reduction='mean')

    with torch.no_grad():
        for batch, ((B1,B2,B3), R, C) in enumerate(zip(Bert_valloader, Resnet_valloader,Category_valloader)):
            R = R.float()
            B1,B2,B3 = B1.to(device), B2.to(device), B3.to(device)
            R,C = R.to(device), C.to(device)
            B_pred = My_model.BERT(B1, B2, B3)
            R_pred = My_model.ResNet(R)
            C_pred = My_model.Classification(R_pred)
            # Compute prediction and loss
            loss = 0.01 * classifier_loss(C_pred,C) + mse_loss(B_pred,R_pred)
            loss = loss.item()
            test_loss += loss

            current += len(R)
            if batch % 5 == 0:
                print(f"loss: {loss:7f}  [{current:5d}/{size:5d}]", end='\r')

        print()

    test_loss /= (batch + 1)
    print(f"{mode} Error: \n Avg loss: {test_loss:8f} \n")

    return test_loss


def train_steps(epochs, history, early_stop, step,
                    PBert_trainloader, NBert_trainloader, Bert_trainloader, Resnet_trainloader, My_model, My_model_optimizer,
                    Bert_valloader, Resnet_valloader, Category_valloader,
                    Category_trainloader, device):
    stop_count = 0
    best_loss = None

    for epoch in range(epochs):
        # timer.start()
        start_time = time()
        print(f"Epoch {epoch+1}\n----------------------------------------")
        history['epoch'].append(epoch + 1)
        if (epoch+1) % 25 == 0:
            torch.save(My_model.state_dict(), f"./baseline/data/model_weight/My_model_pretrain_{step}_{epoch+1}.weights")

        train_history = train_loop(PBert_trainloader, NBert_trainloader, Bert_trainloader, Resnet_trainloader, My_model, My_model_optimizer,
                                Category_trainloader, step, device)
        history['train_loss'].append(train_history)

        end_time = time()
        time_taken = end_time - start_time # time_taken is in seconds
        hours, rest = divmod(time_taken,3600)
        minutes, seconds = divmod(rest, 60)
        print("This took %d hours %d minutes %d seconds" %(hours,minutes,seconds)) 

    return history

def fit(epochs, PBert_trainloader,NBert_trainloader, Resnet_trainloader, device, 
        Bert_trainloader,
        Category_trainloader, Category_valloader, My_model, My_model_optimizer,
        Bert_valloader = None, PBert_valloader = None, NBert_valloader = None, Resnet_valloader = None, early_stop=None):
    '''
    train a model with cpu or gpu device
    '''

    # My_model.load_state_dict(torch.load('./baseline/data/model_weight/My_model_pretrain_1.weights'))

    history = {}
    history['epoch'] = []
    history['train_loss'] = []
    history['val_loss'] = []
    print("Training...")

    # step = 0


    for step in range(1,3):
        print(f'step {step}:')

        if step == 1:
            for i in My_model.BERT.fc1.parameters():
                i.requires_grad=False
    
        history = train_steps(epochs = epochs[step-1], history = history, early_stop = early_stop, step = step,
                PBert_trainloader = PBert_trainloader, NBert_trainloader = NBert_trainloader,
                Bert_trainloader = Bert_trainloader, Resnet_trainloader = Resnet_trainloader, 
                Bert_valloader = Bert_valloader,
                Resnet_valloader = Resnet_valloader,Category_valloader = Category_valloader,
                My_model = My_model, My_model_optimizer = My_model_optimizer,
                Category_trainloader = Category_trainloader, device = device)
        
        torch.save(My_model.state_dict(), f"./baseline/data/model_weight/My_model_pretrain_{step}.weights")


    return history