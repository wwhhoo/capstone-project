import faiss
import json
import torch
from math import log10
from time import time
import numpy as np
# from torch.functional import Tensor
# from torch.nn.functional import mse_loss
import shutil
from torch.nn import TripletMarginLoss,CrossEntropyLoss,MSELoss


def train_loop(PBert_trainloader, NBert_trainloader, Bert_trainloader, Resnet_trainloader, My_model,  My_model_optimizer,
                Category_trainloader, step, device):
    '''
    train a model in an epoch
    '''
    size = len(Resnet_trainloader.dataset)
    current = 0
    train_loss = 0
    My_model.train()

    classifier_loss = CrossEntropyLoss()
    triplet_loss = TripletMarginLoss(margin=0.2, p=2)
    mse_loss = MSELoss(reduction='mean')
    

    for batch, ((PB1,PB2,PB3),(NB1,NB2,NB3),(B1,B2,B3), R, C) in enumerate(zip(PBert_trainloader, NBert_trainloader, Bert_trainloader, Resnet_trainloader, Category_trainloader)):

        R = R.float()
        PB1,PB2,PB3 = PB1.to(device), PB2.to(device), PB3.to(device)
        NB1,NB2,NB3 = NB1.to(device), NB2.to(device), NB3.to(device)
        B1,B2,B3 = B1.to(device), B2.to(device), B3.to(device)
        R,C = R.to(device), C.to(device)
        # Compute prediction and loss

        if step == 1:
            B_pred = My_model.BERT(B1,B2,B3)
            PB_pred = My_model.BERT(PB1,PB2,PB3)
            NB_pred = My_model.BERT(NB1,NB2,NB3)
            loss = triplet_loss(B_pred,PB_pred,NB_pred)
        elif step == 2:
            # loss = classifier_loss(C_pred,C)
            B_pred = My_model.BERT(B1,B2,B3)
            PB_pred = My_model.BERT(PB1,PB2,PB3)
            NB_pred = My_model.BERT(NB1,NB2,NB3)
            C_pred = My_model.Classification(B_pred)
            loss = triplet_loss(B_pred,PB_pred,NB_pred) + classifier_loss(C_pred,C)
        # else:
        #     PB_pred, NB_pred, B_pred, R_pred, C_pred = My_model(PB1, PB2, PB3, NB1, NB2, NB3, B1, B2, B3, R, step)
        #     # print(B_pred.shape)
        #     loss = triplet_loss(R_pred,PB_pred,NB_pred) + 0.01 * classifier_loss(C_pred,C) + mse_loss(B_pred,R_pred)
        #     # loss = 0.4 * triplet_loss(R_pred,PB_pred,NB_pred) + mse_loss(B_pred,R_pred)
        #     # loss = classifier_loss(C_pred,C)#mse_loss(B_pred,R_pred)
        elif step == 3 or step == 4:
            B_pred = My_model.BERT(B1,B2,B3)
            R_pred = My_model.ResNet(R)
            loss = mse_loss(B_pred,R_pred)
        elif step == 5:
            PB_pred, NB_pred, B_pred, R_pred, C_pred = My_model(PB1, PB2, PB3, NB1, NB2, NB3, B1, B2, B3, R, step)
            loss = triplet_loss(R_pred,PB_pred,NB_pred) + 0.01 * classifier_loss(C_pred,C) + mse_loss(B_pred,R_pred)

        # Backpropagation
        My_model_optimizer.zero_grad()
        loss.backward()
        My_model_optimizer.step()

        loss = loss.item()
        train_loss += loss

        current += len(R)
        if batch % 5 == 0:
            print(f"loss: {loss:7f}  [{current:5d}/{size:5d}]", end='\r')
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

def recall(label, faiss_I, top, ad):

    hit_rate = np.zeros(int(log10(ad)+1), dtype=float)
    for top_num in range(top):
        
        hit = np.where(faiss_I == label)
        # print(hit)

        if hit[0].size > 0:
            # for i in range(hit[0].size):
            if (hit[0][0] < 1 and top == 1):
                hit_rate[0:int(log10(ad)+1)] = 1
            elif (hit[0][0] < 5 and top <= 10 ):
                hit_rate[1:int(log10(ad)+1)] = 1
            elif (hit[0][0] < 10 ):
                hit_rate[2:int(log10(ad)+1)] = 1
            elif (hit[0][0] < 1000000 ):
                hit_rate[3] = 1
    return hit_rate

def save_result(q_num, DS_ID, label, faiss_I, order, query, dataset, nums_save_file, mode):
    the_candidate = mode.split('_')[2]
    the_query = mode.split('_')[0]
    if the_query == "image":
        source = query[q_num]
        destination = f"D:/James/口試/圖表/compare/{mode}/{q_num}_pos_query_{label}.png"
        shutil.copyfile(source, destination)
    elif the_query == "text":
        text_file = open(f"D:/James/口試/圖表/compare/{mode}/{q_num}_pos_query_{label}.txt", "w",encoding="utf-8")
        text_file.write(query[q_num])
        text_file.close()
        # print(query[q_num])
        # print(text_file.append("dsad"))
        # np.savetxt(f"D:/James/口試/圖表/compare/{mode}/{nums_save_file}_pos_query_{label}.txt",text_file.append(query[q_num]))

    for i in range (5):
        if the_candidate == "image":
            source = dataset[order[i]]
            destination = f"D:/James/口試/圖表/compare/{mode}/{q_num}_pos_dataset_{i}_{faiss_I[i]}.png"
            shutil.copyfile(source, destination)
        elif the_candidate == "text":
            text_file = open(f"D:/James/口試/圖表/compare/{mode}/{q_num}_pos_dataset_{i}_{faiss_I[i]}.txt", "w",encoding="utf-8")
            text_file.write(dataset[order[i]])
            text_file.close()
            # np.savetxt(f"D:/James/口試/圖表/compare/{mode}/{nums_save_file}_pos_dataset_{faiss_I[i]}_{i}.txt",dataset[order[i]])
    return 0

def predictions(label, faiss_I, top, ad, DS_ID,search_ID, order, q_num, query_data, cand_dataset, nums_save_file, mode):

    hit_rate = np.zeros(int(log10(ad)+1), dtype=float)

    for top_num in range(top):

        hit = np.where(faiss_I == label)
        if hit[0].size > 0:
            num = np.where(faiss_I[:5] == label)
            # if num[0].size > 2:
                # print(f'Query {q_num} : {search_ID}')
                # print(f'Candidate : {DS_ID[:5]}')
                # print(f'Query class : {label}')
                # print(f'Candidate class : {faiss_I[:5]}')
                # print(f'order : {order[:5]}')
                # nums_save_file +=1
                # save_result(q_num,DS_ID[:5],label,faiss_I[:5],order[:5],query_data,cand_dataset,nums_save_file,mode)
            # print(hit[0])
            for i in range(hit[0].size):
                if (hit[0][i] < 1 and top == 1):
                    hit_rate[0:int(log10(ad)+1)] += 1
                    # num = np.where(faiss_I[:5] == label)
                    # if num[0].size > 2:
                    #     print(f'Query {q_num} : {search_ID}')
                    #     print(f'Candidate : {DS_ID[:5]}')
                    #     print(f'Query class : {label}')
                    #     print(f'Candidate class : {faiss_I[:5]}')
                    #     print(f'order : {order[:5]}')


                elif (hit[0][i] < 5 and top <= 10 ):
                    hit_rate[1:int(log10(ad)+1)] += 1
                elif (hit[0][i] < 10 ):
                    hit_rate[2:int(log10(ad)+1)] += 1
                    # print(f'Query {q_num} : {search_ID}')
                    # print(f'Candidate : {DS_ID[:5]}')
                    # print(f'Query class : {label}')
                    # print(f'Candidate class : {faiss_I[:5]}')
                    # print(f'order : {order[:5]}')
                    # save_result(q_num,DS_ID[:5],label,faiss_I[:5],order[:5],query_data,cand_dataset,nums_save_file,mode)
                elif (hit[0][i] < 1000000 ):
                    hit_rate[3] += 1
    return hit_rate

# Same class
def compute_results(order,mode):

    dic = "./baseline/data/dic_for_class.json"
    id_to_class = json.load(open(dic))
    dic = "./baseline/data/rough_detail_class.json"
    # rough_detail_class = json.load(open(dic))
    recall_score = np.zeros(4)
    predictions_score = np.zeros(4)
    top = 1
    ad = 1000
    rem_ID = []
    search_ID = []
    get_class = []
    query_class = np.load('./baseline/data/sorted/test/test_imageID_order.npy')
    dataset_class  = np.load('./baseline/data/sorted/Total/Total_imageID_order.npy')
    #####
    the_dataset = mode.split('_')[2]
    the_query = mode.split('_')[0]
    if the_dataset == "image":
        cand_dataset = np.load("./baseline/data/sorted/Total/Total_dataset_image.npy")
    elif the_dataset == "text":
        cand_dataset = np.load("./baseline/data/sorted/Total/Total_ori_text.npy")
    if the_query == "image":
        query_data = np.load("./baseline/data/sorted/test/test_dataset_image.npy")
    elif the_query == "text":
        query_data = np.load("./baseline/data/sorted/test/test_ori_text.npy")
    # print(cand_dataset)
    #####
    nums_save_file = 0
    for i in range(len(order)):
        get_class = dataset_class[order[i]]
        rem_ID = dataset_class[order[i]]

        for j in range(len(get_class)):
            # get_class[j] = rough_detail_class[id_to_class[get_class[j].split('-')[0]]]
            get_class[j] = id_to_class[get_class[j].split('-')[0]][:2]
        search_ID = query_class[i]
        # print(id_to_class[query_class[i].split('-')[0]])
        # query_class[i] = rough_detail_class[id_to_class[query_class[i].split('-')[0]]]
        query_class[i] = id_to_class[query_class[i].split('-')[0]][:2]
        predictions_score += predictions(query_class[i], get_class, top, ad,rem_ID , search_ID, order[i], i,query_data,cand_dataset,nums_save_file,mode)
        recall_score += recall(query_class[i], get_class, top, ad)
        
    print(recall_score)
    print(predictions_score)

    recall_score = (recall_score/(len(order)*top))
    predictions_score = (predictions_score/(len(order)*top))
    

    return recall_score, predictions_score


@torch.no_grad()
def predict(Bert_testloader, Resnet_testloader, Bert_Totalloader, Resnet_Totalloader,
            My_model, device, mode):
    '''
    model prediction of dataset with cpu or gpu device
    '''
    # My_model.load_state_dict(torch.load('./baseline/data/model_weight/avoid/My_model_pretrain_2_100.weights'))
    My_model.load_state_dict(torch.load('./baseline/data/model_weight/My_model_5_200.weights'))

    predictions = []
    Dataset_predict = []
    Query_predict = []
    Category_predict = []
    My_model.eval()
    the_dataset = mode.split('_')[2]
    the_query = mode.split('_')[0]
    # Dataset
    if the_dataset == "image":
        D_dataloader = Resnet_Totalloader
        size = len(D_dataloader.dataset)
        current = 0
        print("Load Resnet data:")
        for batch, (R) in enumerate(D_dataloader):
            R = R.float()
            R = R.to(device)
            R_pred = My_model.ResNet(R)
            
            Dataset_predict.extend(R_pred.cpu().data.numpy())
            
            current += len(R)
            if batch % 5 == 0:
                print(f"Progress: [{current:5d}/{size:5d}]", end='\r')
    elif the_dataset == "text":
        D_dataloader = Bert_Totalloader
        size = len(D_dataloader.dataset)
        current = 0
        print("Load Bert data:")
        for batch, ((B1,B2,B3)) in enumerate(D_dataloader):
            B1,B2,B3 = B1.to(device), B2.to(device), B3.to(device)
            B_pred = My_model.BERT(  input_ids=B1, 
                            token_type_ids=B2, 
                            attention_mask=B3)

            Dataset_predict.extend(B_pred.cpu().data.numpy())

            current += len(B1)
            if batch % 5 == 0:
                print(f"Progress: [{current:5d}/{size:5d}]", end='\r')
    print()

    # Query
    if the_query == "image":
        Q_dataloader = Resnet_testloader
        size = len(Q_dataloader.dataset)
        current = 0
        print("Load Resnet data:")
        for batch, (R) in enumerate(Q_dataloader):
            R = R.float()
            R = R.to(device)
            R_pred = My_model.ResNet(R)
            C_pred = My_model.Classification(R_pred)
            Query_predict.extend(R_pred.cpu().data.numpy()) 
            Category_predict.extend(C_pred.cpu().data.numpy())
            current += len(R)
            if batch % 5 == 0:
                print(f"Progress: [{current:5d}/{size:5d}]", end='\r')
    elif the_query == "text":
        Q_dataloader = Bert_testloader
        size = len(Q_dataloader.dataset)
        current = 0
        print("Load Bert data:")
        for batch, ((B1,B2,B3)) in enumerate(Q_dataloader):
            B1,B2,B3 = B1.to(device), B2.to(device), B3.to(device)
            B_pred = My_model.BERT(  input_ids=B1, 
                            token_type_ids=B2, 
                            attention_mask=B3)
            C_pred = My_model.Classification(B_pred)
            Query_predict.extend(B_pred.cpu().data.numpy())
            Category_predict.extend(C_pred.cpu().data.numpy())
            current += len(B1)
            if batch % 5 == 0:
                print(f"Progress: [{current:5d}/{size:5d}]", end='\r')
    print()
    print(mode)
    ######################################
    category_label = np.load('./baseline/data/sorted/test/test_dataset_category.npy')
    dic = "./baseline/data/detail_rough_class.json"
    detail_rough_class = json.load(open(dic))
    rough_acc_num = np.zeros((len(category_label)), dtype = np.uint8)
    detail_acc_num = np.zeros((len(category_label)), dtype = np.uint8)
    # max_pos = np.argmax(Category_predict,axis=1)
    
    # print(Category_predict)
    ad_num = np.zeros((3,len(category_label)), dtype = np.uint8)
    for i in range (3):
        max_pos = np.argmax(Category_predict,axis=1)
        for j in range (len(Category_predict)):
            # print(max_pos[j])
            ad_num[i,j] = max_pos[j]
            Category_predict[j][max_pos[j]] = 0
    
    for ad in range (3):
        count = 0
        big_count = 0
        for i in range (len(Category_predict)):
            if detail_rough_class[str(category_label[i])][:2] == detail_rough_class[str(ad_num[ad][i])][:2]:
                rough_acc_num[i] = 1 
                # big_count+=1
                if category_label[i] == ad_num[ad][i]:
                    detail_acc_num[i] = 1
                    # count+=1
        # for i in range (len(Category_predict)):
        #     category_label[i] = detail_rough_class[str(category_label[i])][:2]
        #     max_pos[i] = detail_rough_class[str(max_pos[i])][:2]
        # print(category_label)
        # print(max_pos)
        for i in range (len(Category_predict)):
            if rough_acc_num[i] == 1:
                big_count +=1
            if detail_acc_num[i] == 1:
                count +=1
        # print(ad_num[ad])
        print(f"@{ad+1} rough class : {big_count}")
        print(f"@{ad+1} detail class: {count}")
    ######################################
    FlatIndex = faiss.IndexFlatL2(768)
    k = 1000
    
    Dataset_predict = np.asarray(Dataset_predict)
    Query_predict = np.asarray(Query_predict)
    FlatIndex.add(Dataset_predict)
    D,I = FlatIndex.search(Query_predict,k)
    if mode == "image_to_image" or mode == "text_to_text":
        recall, predictions = compute_results(I[:,1:],mode)
    else:
        recall, predictions = compute_results(I,mode)
    recall, predictions = compute_results(I,mode)
    
    np.save("./baseline/data/result/predict_class.npy",ad_num)
    np.savetxt("./baseline/data/result/predict_class.txt",ad_num)

    return recall, predictions 

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
            torch.save(My_model.state_dict(), f"./baseline/data/model_weight/My_model_{step}_{epoch+1}.weights")

        train_history = train_loop(PBert_trainloader, NBert_trainloader, Bert_trainloader, Resnet_trainloader, My_model, My_model_optimizer,
                                Category_trainloader, step, device)
        history['train_loss'].append(train_history)

        # if Bert_valloader:
        #     val_history = test_loop(Bert_valloader, Resnet_valloader, My_model, Category_valloader, device, 'Validation')
        #     history['val_loss'].append(val_history)

        #     if best_loss and best_loss < history['val_loss'][-1]:
        #         stop_count += 1

        #     else:
        #         My_model_best_weights = My_model.state_dict()
        #         best_loss = history['val_loss'][-1]
        #         stop_count = 0
            
        #     if stop_count == early_stop:
        #         My_model.load_state_dict(My_model_best_weights)
        #         break

        end_time = time()
        time_taken = end_time - start_time # time_taken is in seconds
        hours, rest = divmod(time_taken,3600)
        minutes, seconds = divmod(rest, 60)
        print("This took %d hours %d minutes %d seconds" %(hours,minutes,seconds)) 
    # if stop_count:
    #     My_model.load_state_dict(My_model_best_weights)
        # timer.finish()

        # print(f"Time: {timer.elapsed_time:.2f} sec, ETA: {timer.ETA}\n")


    return history

def fit(epochs, PBert_trainloader,NBert_trainloader, Resnet_trainloader, device, 
        Bert_trainloader,
        Category_trainloader, Category_valloader, My_model, My_model_optimizer,
        Bert_valloader = None, PBert_valloader = None, NBert_valloader = None, Resnet_valloader = None, early_stop=None):
    '''
    train a model with cpu or gpu device
    '''

    My_model.load_state_dict(torch.load('./baseline/data/model_weight/My_model_5_150.weights'))
    # print(Bert)
    history = {}
    history['epoch'] = []
    history['train_loss'] = []
    history['val_loss'] = []
    print("Training...")

    step = 0


    for step in range(1,6):
        print(f'step {step}:')

        if step == 1:
            for i in My_model.ResNet.parameters():
                i.requires_grad=False
        elif step == 3:
            for i in My_model.BERT.parameters():
                i.requires_grad=True
            for i in My_model.Classification.parameters():
                i.requires_grad=False
            for i in My_model.ResNet.parameters():
                i.requires_grad=False
        elif step == 4:
            for i in My_model.BERT.parameters():
                i.requires_grad=False
            for i in My_model.ResNet.parameters():
                i.requires_grad=True       
        elif step == 5:
            for i in My_model.BERT.parameters():
                i.requires_grad=True
            for i in My_model.Classification.parameters():
                i.requires_grad=True
            for i in My_model.ResNet.parameters():
                i.requires_grad=True

            # pre_model = torch.load('./baseline/data/model_weight/My_model_pretrain_2_50.weights')
            # My_model_dic = My_model.state_dict()
            # state_dict = {'Classification.fc1.weight': pre_model['Classification.fc1.weight']}
            # My_model_dic.update(state_dict)
            # My_model.load_state_dict(My_model_dic)
    
        history = train_steps(epochs = epochs[step-1], history = history, early_stop = early_stop, step = step,
                PBert_trainloader = PBert_trainloader, NBert_trainloader = NBert_trainloader,
                Bert_trainloader = Bert_trainloader, Resnet_trainloader = Resnet_trainloader, 
                Bert_valloader = Bert_valloader,
                Resnet_valloader = Resnet_valloader,Category_valloader = Category_valloader,
                My_model = My_model, My_model_optimizer = My_model_optimizer,
                Category_trainloader = Category_trainloader, device = device)
        
        # torch.save(My_model.state_dict(), f"./baseline/data/model_weight/My_model_{step}.weights")


    return history

@torch.no_grad()
def evaluate(Bert_testloader, Resnet_testloader, Bert_Totalloader, Resnet_Totalloader, 
            My_model, device, mode):
    '''
    evaluate a model with cpu or gpu device
    '''

    recall_score, predictions_score = predict(Bert_testloader, Resnet_testloader, Bert_Totalloader, Resnet_Totalloader,
                                                My_model, device, mode)

    print(f"Recall: {recall_score}")
    np.savetxt("./baseline/data/result/recall.txt", recall_score)

    print(f"predictions: {predictions_score}")
    np.savetxt("./baseline/data/result/predictions.txt", predictions_score)
    

    return recall_score, predictions_score