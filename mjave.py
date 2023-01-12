import torch
import torch.nn as nn
from transformers import BertModel
from resnet_JAVE import resnet152

# config = {
#     "txt_hidden_size": 768, # hidden dim of pre-trained bert
#     "img_hidden_size": 2048, # hidden dim of pre-trained resnet(last conv layer)
#     "img_global_size": 2048, # hidden dim of pre-trained resnet(last pooling layer)
#     "img_block_num": 49, # # num of regional image features  (7×7=49)
#     "attn_size": 200, # hidden dim in attention
#     "batch_size": 128, # batch size
#     "dropout_prob": 0, # probability of dropout layers,
#     "vocab_size_label": 10,
#     "vocab_size_bio": 2,
#     "device": DEVICE,
#     "test": False
# }

class M_JAVE(nn.Module):
    def __init__(self, bert_name, config):
        super(M_JAVE, self).__init__()
        self.txt_hidden_size = config['txt_hidden_size']
        self.img_hidden_size = config['img_hidden_size']
        self.attention_size = config['attn_size']
        self.dropout_prob = config['dropout_prob']
        self.device = config['device']
        self.vocab_size_label = config['vocab_size_label']
        self.vocab_size_bio = config['vocab_size_bio']
        
        self.test = config['test']
        
        self.bert = BertModel.from_pretrained(bert_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        print('load pretrained BERT')

        self.pre_resnet = resnet152()
        self.pre_resnet.load_state_dict(torch.load('./pth/resnet152-b121ed2d.pth'))
        print('load pretrained resnet152')
        
        for param in self.pre_resnet.parameters():
            param.requires_grad = False

        '''
        txt-txt
        '''
        
        self.scale = torch.sqrt(torch.FloatTensor([self.attention_size])).to(self.device)
        
        self.txt_fc_q = nn.Linear(self.txt_hidden_size, self.attention_size)
        self.txt_fc_k = nn.Linear(self.txt_hidden_size, self.attention_size)
        self.txt_fc_v = nn.Linear(self.txt_hidden_size, self.attention_size)
        self.txt_fc_o = nn.Linear(self.attention_size, self.attention_size)
        self.txt_dropout = nn.Dropout(self.dropout_prob)
        
        '''
        img-txt
        '''
        
        self.img_fc_q = nn.Linear(self.txt_hidden_size, self.attention_size)
        self.img_fc_k = nn.Linear(self.img_hidden_size, self.attention_size)
        self.img_fc_v = nn.Linear(self.img_hidden_size, self.attention_size)
        self.img_fc_o = nn.Linear(self.attention_size, self.img_hidden_size)
        
        '''
        global gate
        '''
        
        self.global_gate_txt = nn.Linear(self.txt_hidden_size, 1) # for text
        self.global_gate_imgG = nn.Linear(self.img_hidden_size, 1) # for imageG
        self.sigmoid = nn.Sigmoid()
        emb = torch.empty(1,1)
        nn.init.xavier_uniform_(emb)
        self.bias = nn.Parameter(torch.Tensor(emb))
        
        '''
        label projection
        '''
        
        self.linear3 = nn.Linear(self.txt_hidden_size, self.attention_size)
        self.dropout3 = nn.Dropout(self.dropout_prob)
        self.linear4 = nn.Linear(self.attention_size, self.attention_size)
        self.dropout4 = nn.Dropout(self.dropout_prob)
        self.linear5 = nn.Linear(self.img_hidden_size, self.attention_size)
        self.dropout5 = nn.Dropout(self.dropout_prob)

        self.label_linear = nn.Linear(self.attention_size, self.vocab_size_label)
        
        '''
        regional gate
        '''
        
        self.linear6 = nn.Linear(self.vocab_size_label, 1)
        self.linear7 = nn.Linear(self.img_hidden_size, 1)
        
        '''
        sequence projection
        '''
        
        self.linear8 = nn.Linear(self.txt_hidden_size, self.attention_size)
        self.dropout8 = nn.Dropout(self.dropout_prob)
        self.linear9 = nn.Linear(self.attention_size, self.attention_size)
        self.dropout9 = nn.Dropout(self.dropout_prob)
        self.linear10 = nn.Linear(self.vocab_size_label, self.attention_size)
        self.dropout10 = nn.Dropout(self.dropout_prob)
        
        self.logit_linear = nn.Linear(self.attention_size, self.vocab_size_bio)
        
        
    def forward(
        self, 
        input_ids, 
        images, 
        token_type_ids=None, 
        attention_mask=None, 
        use_image_global=True, 
        use_labels=True, 
        use_images_regional=True,
        use_images_global=True
    ):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        img_block_num = 49
        
        bert_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        inputs_seq_embedded = bert_output[0] # (batch, seq_len, text_hidden_size=768)
        inputs_seq_embeddedG = bert_output[1] # (batch, text_hidden_size=768)
        if self.test:
            print('inputs_seq_embedded: ', inputs_seq_embedded)
            print('inputs_seq_embeddedG: ', inputs_seq_embeddedG)
        
        
        inputs_img_embedded, inputs_img_embeddedG = self.pre_resnet(images) # (batch, image_hidden_size=2048, 7, 7) / (batch, image_hidden_size=2048, 1, 1)
        inputs_img_embedded = inputs_img_embedded.view(batch_size, self.img_hidden_size, 7*7).transpose(2, 1).contiguous() # (batch, img_block_num=49, image_hidden_size=2048)
        
        inputs_img_embeddedG = inputs_img_embeddedG.view(batch_size, self.img_hidden_size).contiguous() # (batch, image_hidden_size=2048)
        
        '''
        txt-txt
        '''
        
        txt_Q = self.txt_fc_q(inputs_seq_embedded) # Q = (batch, query_len=seq_len, attn_hidden_size)
        txt_K = self.txt_fc_k(inputs_seq_embedded) # K = (batch, key_len=seq_len, attn_hidden_size)
        txt_V = self.txt_fc_v(inputs_seq_embedded) # V = (batch, value_len=seq_len, attn_hidden_size)
        
        txt_Q = txt_Q.view(batch_size, -1, 1, self.attention_size).permute(0, 2, 1, 3).contiguous() # (batch, n_heads, query_len, head_dim)
        txt_K = txt_K.view(batch_size, -1, 1, self.attention_size).permute(0, 2, 1, 3).contiguous() # (batch, n_heads, key_len, head_dim)
        txt_V = txt_V.view(batch_size, -1, 1, self.attention_size).permute(0, 2, 1, 3).contiguous() # (batch, n_heads, value_len, head_dim)
        
        if self.test:
            print('txt_Q: ', txt_Q)
            print('txt_K: ', txt_K)
            print('txt_V: ', txt_V)
                
        txt_QK = torch.div(torch.matmul(txt_Q, txt_K.permute(0, 1, 3, 2).contiguous()), self.scale) # (batch, n_heads, query_len, key_len)
        
        if self.test:
            print('txt_QK: ', txt_QK)
        
        # softmax + dropout
        # if attention_mask is not None:
        #     txt_QK = txt_QK.masked_fill(attention_mask == 0, -1e10)
        # txt_attention = torch.softmax(txt_QK, dim = -1)
        # hiddens_txt = torch.matmul(self.txt_dropout(txt_attention), txt_V) # (batch, n_heads, query_len, head_dim)
        
        hiddens_txt = torch.matmul(txt_QK, txt_V) # (batch, n_heads, query_len, head_dim)
        
        hiddens_txt = hiddens_txt.permute(0, 2, 1, 3).contiguous() # (batch, query_len, n_heads, head_dim)
        
        hiddens_txt = hiddens_txt.view(batch_size, -1, self.attention_size).contiguous() # (batch, seq_len, attn_hidden_size)
        
        # hiddens_txt = self.txt_fc_o(hiddens_txt) # (batch, query_len, attn_hidden_size)
        if self.test:
            print('hiddens_txt: ', hiddens_txt)
        
        if use_image_global:
            '''
            img-txt
            '''
            img_Q = self.img_fc_q(inputs_seq_embedded) # Q = (batch, query_len=seq_len, attn_hidden_size)
            img_K = self.img_fc_k(inputs_img_embedded) # K = (batch, key_len=img_block_num, attn_hidden_size)
            img_V = self.img_fc_v(inputs_img_embedded) # V = (batch, value_len=img_block_num, attn_hidden_size)
            
            img_Q = img_Q.view(batch_size, -1, 1, self.attention_size).permute(0, 2, 1, 3).contiguous() # (batch, n_heads, query_len, head_dim)
            img_K = img_K.view(batch_size, -1, 1, self.attention_size).permute(0, 2, 1, 3).contiguous() # (batch, n_heads, key_len, head_dim)
            img_V = img_V.view(batch_size, -1, 1, self.attention_size).permute(0, 2, 1, 3).contiguous() # (batch, n_heads, value_len, head_dim)

            img_QK = torch.div(torch.matmul(img_Q, img_K.permute(0, 1, 3, 2).contiguous()), self.scale) # (batch, n_heads, query_len, key_len)
            
            # softmax + dropout
            # if attention_mask is not None:
            #     img_QK = img_QK.masked_fill(attention_mask == 0, -1e10)
            # img_attention = torch.softmax(img_QK, dim = -1)
            # hiddens_cross_modality = torch.matmul(self.txt_dropout(img_attention), img_V) # (batch, n_heads, query_len, head_dim)
            
            hiddens_cross_modality = torch.matmul(img_QK, img_V) # (batch, n_heads, query_len, head_dim)
            hiddens_cross_modality = hiddens_cross_modality.permute(0, 2, 1, 3).contiguous() # (batch, query_len, n_heads, head_dim)
            hiddens_cross_modality = hiddens_cross_modality.view(batch_size, -1, self.attention_size).contiguous() # (batch, seq_len, attn_hidden_size)
            
            '''
            global gate
            '''
            d1 = self.global_gate_txt(inputs_seq_embedded) # (batch, seq_len, 1)
            d1 = torch.squeeze(d1, -1) # (batch, seq_len)
            d2 = self.global_gate_imgG(inputs_img_embeddedG) # (batch, 1)
            g1 = self.sigmoid(d1 + d2 + self.bias) # (batch, seq_len)
            
            hiddens_img1 = torch.mul(torch.unsqueeze(g1, dim=2), hiddens_cross_modality) # (batch, seq_len, attention_size)
        else:
            hiddens_img1 = torch.zeros(batch_size, seq_len, self.attn_size, device=self.device)
        
        if self.test:
            print('hiddens_img1: ', hiddens_img1)
        
        hiddens_mm = hiddens_txt + hiddens_img1 # (batch, seq_len, attention_size)

        if self.test:
            print('hiddens_mm: ', hiddens_mm)
            
        if use_labels:
            '''
            label projection
            '''
            d3 = self.linear3(torch.sum(inputs_seq_embedded, dim=1)) # (batch, attention_size)
            d3 = self.dropout3(d3)
            d4 = self.linear4(torch.sum(hiddens_mm, dim=1)) # (batch, attnetion_size)
            d4 = self.dropout4(d4)
            d5 = self.linear5(inputs_img_embeddedG) # (batch, attention_size)
            d5 = self.dropout5(d5)
            logits_label = self.label_linear(d3 + d4 + d5)
            preds_label = self.sigmoid(logits_label) # (batch, vocab_size_label)
        else:
            preds_label = torch.zeros(batch_size, self.vocab_size_label, device=self.device)
        
        if use_images_regional and use_images_global and use_labels:
            '''
            regional gate
            '''
            d6 = self.linear6(preds_label) # (batch, 1)
            d7 = self.linear7(inputs_img_embedded) # (batch, img_block_num=49, 1)
            d7 = torch.squeeze(d7, dim=-1)
            g2 = self.sigmoid(d6 + d7) # (batch, img_block_num=49)
            hidden_img2 = torch.matmul(torch.squeeze(img_QK, dim=1) * torch.unsqueeze(g2, dim=1), img_V)
            hidden_img2 = hidden_img2.permute(0, 2, 1, 3).contiguous() # (batch, query_len, n_heads, head_dim)
            hidden_img2 = hidden_img2.view(batch_size, -1, self.attention_size).contiguous() # (batch, seq_len, attn_hidden_size)
        else:
            hidden_img2 = torch.zeros(batch_size, seq_len, self.attention_size, device=self.device)
            
        
        '''
        sequence projection
        '''
        d8 = self.linear8(inputs_seq_embedded) # (batch, seq_len, attn_hidden_size)
        d8 = self.dropout8(d8)
        d9 = self.linear9(hiddens_mm) # (batch, seq_len, attn_hidden_size)
        d9 = self.dropout9(d9)
        d10 = self.linear10(preds_label) # (batch, attn_hidden_size)
        d10 = self.dropout10(d10)
        d10 = d10.unsqueeze(1) # (batch, 1, attn_hidden_size)
        logits_seq = self.logit_linear(d8 + d9 + d10 + hidden_img2) # (batch, seq_len, vocab_size_bio)
        if self.test:
            print('d8: ', d8)
            print('d9: ', d9)
            print('d10: ', d10)
            
            print('logits_seq: ', logits_seq)
        
        preds_seq = torch.softmax(logits_seq, dim=2)
        
        outputs = [preds_seq, preds_label]
        
        return outputs
