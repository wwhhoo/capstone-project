import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
from torchvision.models import resnet34
from transformers import  BertForSequenceClassification#BertForTokenClassification,BertForSequenceClassification
# from torchvision.transforms import Resize, ToTensor

class My_Text_Model(nn.Module):#nn.Module
    def __init__(self, NUM_LABELS = 1024, outputs_dims = 768): 
        super(My_Text_Model,self).__init__()
        PRETRAINED_MODEL_NAME = "bert-base-chinese"
        self.bert = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels = NUM_LABELS)
        # for i in self.bert.parameters():
        #     i.requires_grad=False
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(NUM_LABELS, outputs_dims)
        self.layer_norm = nn.LayerNorm(outputs_dims)
    def forward(self, input_ids, token_type_ids=None, attention_mask=None): 
        text = self.bert(input_ids, token_type_ids, attention_mask)
        text = self.dropout1(text[0])
        text = self.fc1(text)
        text = self.layer_norm(text)

        return text

class My_Image_model(nn.Module):#torch.nn.Module
    def __init__(self, outputs_dims = 768):
        super(My_Image_model, self).__init__()
        self.image_model = resnet34(pretrained=True)
        #self.resnet = resnet34(pretrained=True)
        self.image_model.fc = nn.Linear(512,outputs_dims)
        self.layer_norm = nn.LayerNorm(outputs_dims)
    
    def forward(self, image):
        # image = self.resnet(image)
        image = self.image_model(image)
        image = self.layer_norm(image)

        return image

class Classification_model(nn.Module):
    def __init__(self, outputs_dims = 158):
        super(Classification_model, self).__init__()
        fc_features = 768
        self.fc1 = nn.Linear(fc_features,outputs_dims)
        self.softmax = torch.nn.Softmax(1)
    
    def forward(self, data):
        data = self.fc1(data)
        data = self.softmax(data)

        return data

class Propose_model(nn.Module):
    def __init__(self):
        super(Propose_model, self).__init__()
        self.BERT = My_Text_Model()
        self.ResNet = My_Image_model()
        self.Classification = Classification_model()

        self.softmax = torch.nn.Softmax(1)
    
    def forward(self, Bert_positive_token, Bert_positive_segment, Bert_positive_mask, Bert_negative_token, Bert_negative_segment, Bert_negative_mask, 
                Bert_mse_token, Bert_mse_segment, Bert_mse_mask,
                ResNet_data, step):
        PB_pred = self.BERT(  input_ids    = Bert_positive_token, 
                            token_type_ids = Bert_positive_segment, 
                            attention_mask = Bert_positive_mask)

        NB_pred = self.BERT(  input_ids    = Bert_negative_token, 
                            token_type_ids = Bert_negative_segment, 
                            attention_mask = Bert_negative_mask)
        B_pred =  self.BERT(  input_ids    = Bert_mse_token, 
                            token_type_ids = Bert_mse_segment, 
                            attention_mask = Bert_mse_mask)
        R_pred = self.ResNet(ResNet_data)

        if step == 2 :
            C_pred = self.Classification(PB_pred)
        else:
            C_pred = self.Classification(R_pred)



        return PB_pred, NB_pred, B_pred, R_pred, C_pred
