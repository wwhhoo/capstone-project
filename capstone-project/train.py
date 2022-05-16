from lib.coreFunc import fit
from lib.coreFunc_sec import fit as sec_fit
from lib.my_dataset import TextDataset, ImageDataset, create_mini_batch, ClassifierDataset
from lib.my_model import Propose_model
from torch.utils.data import DataLoader
import torch

if __name__ == '__main__':
    
    
    data_path = './baseline/data/sorted'
    BATCH_SIZE = 4
    # set dataloader
    file = "train"
    PBert_train_data = TextDataset(data_path,file,"positive")#positive
    NBert_train_data = TextDataset(data_path,file,"negative")
    Bert_train_data = TextDataset(data_path,file,"mse")
    Resnet_train_data = ImageDataset(data_path,file,'pretrain')#pretrain
    Category_train_data = ClassifierDataset(data_path,file,"train")
    PBert_trainloader = DataLoader(PBert_train_data, batch_size=BATCH_SIZE,num_workers = 2,
                            collate_fn=create_mini_batch)
    NBert_trainloader = DataLoader(NBert_train_data, batch_size=BATCH_SIZE,num_workers = 2,
                            collate_fn=create_mini_batch)
    Bert_trainloader = DataLoader(Bert_train_data, batch_size=BATCH_SIZE,num_workers = 2,
                            collate_fn=create_mini_batch)
    Resnet_trainloader = DataLoader(Resnet_train_data, batch_size=BATCH_SIZE,num_workers = 2)
    Category_trainloader = DataLoader(Category_train_data, batch_size=BATCH_SIZE,num_workers = 2)


    file = "validation"
    Bert_val_data = TextDataset(data_path,file,"mse")
    Resnet_val_data = ImageDataset(data_path,file,'validation')
    Category_val_data = ClassifierDataset(data_path,file,"validation")
    Bert_valloader = DataLoader(Bert_val_data, batch_size=BATCH_SIZE,num_workers = 2,
                            collate_fn=create_mini_batch)
    Resnet_valloader = DataLoader(Resnet_val_data, batch_size=BATCH_SIZE,num_workers = 2)
    Category_valloader = DataLoader(Category_val_data, batch_size=BATCH_SIZE,num_workers = 2)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    print('========================================')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    # assign model
    My_model = Propose_model()
    My_model = My_model.to(device, dtype=torch.float)

    print('========================================')

    My_model_optimizer = torch.optim.Adam(My_model.parameters(), lr=1e-5)

    # Training
    # sec_fit(epochs = [200,200], PBert_trainloader = PBert_trainloader, NBert_trainloader = NBert_trainloader, Resnet_trainloader = Resnet_trainloader, 
    #                                                 Resnet_valloader = Resnet_valloader,
    #                                                 Bert_trainloader = Bert_trainloader, Bert_valloader = Bert_valloader,
    #                                                 Category_trainloader = Category_trainloader, Category_valloader = Category_valloader,
    #                                                 My_model = My_model, My_model_optimizer = My_model_optimizer,
    #                                                 device = device, early_stop = None)
    file = "train"
    Resnet_train_data = ImageDataset(data_path,file,'train')#pretrain
    Resnet_trainloader = DataLoader(Resnet_train_data, batch_size=BATCH_SIZE,num_workers = 2)
    history = fit(epochs = [200,200,100,100,200], PBert_trainloader = PBert_trainloader, NBert_trainloader = NBert_trainloader, Resnet_trainloader = Resnet_trainloader, 
                                                    Resnet_valloader = Resnet_valloader,
                                                    Bert_trainloader = Bert_trainloader, Bert_valloader = Bert_valloader,
                                                    Category_trainloader = Category_trainloader, Category_valloader = Category_valloader,
                                                    My_model = My_model, My_model_optimizer = My_model_optimizer,
                                                    device = device, early_stop = None)
    print('========================================')