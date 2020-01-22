
import torch.nn as nn
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, embedding_dim = None,num_emd = None, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
#        self.embedding = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=num_emd, padding_idx=0)
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
#        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x, h=None):
        # out = self.embedding(x)
#         print(f'before sum {out.shape}')
#         print(out, length)
        # out = out.sum(dim=-2)
#         print(f'before packing {out.shape}')
        # out = torch.nn.utils.rnn.pack_padded_sequence(out, lengths=length, batch_first=True, enforce_sorted=True)
#         print(f'after packing {out.data.shape}')
        # print(f'x.shape  {x.shape}\n\n')
        if h is not None:   
          out, h = self.gru(x, h)
        else:
          out, h = self.gru(x)  
        # out, out_lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_output,batch_first=True)
#         print(f'after unpacking {out.shape} {out_lengths.shape}')
        # print(f'out.shape  {out.shape}\n\n')
        out = self.fc(out[:,-1,:])

        # out=self.softmax(out)
        return out,h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

class BertForMultiLable(BertPreTrainedModel):
    def __init__(self, config):

        super(BertForMultiLable, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier = GRUNet(1, hidden_dim = 356, output_dim = config.num_labels, n_layers = 3 ) 
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,head_mask=None, hidden_state = None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids,attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]
        # print(f'pooled_output.shape {pooled_output.shape} \n\n')
        pooled_output = pooled_output.unsqueeze(2)
        # pooled_output = self.dropout(pooled_output)
        # print(f'pooled_output.shape {pooled_output.shape} \n\n')

        if hidden_state is None:
          logits, h = self.classifier(pooled_output)
        else:
          logits, h = self.classifier(pooled_output, hidden_state)  
        # logits = self.classifier(pooled_output)  
        # print(f'logits.shape  {logits.shape}   h.shape  {h.shape}\n\n')
        return logits, h
        # return logits   

    def unfreeze(self,start_layer,end_layer):
        def children(m):
            return m if isinstance(m, (list, tuple)) else list(m.children())
        def set_trainable_attr(m, b):
            m.trainable = b
            for p in m.parameters():
                p.requires_grad = b
        def apply_leaf(m, f):
            c = children(m)
            if isinstance(m, nn.Module):
                f(m)
            if len(c) > 0:
                for l in c:
                    apply_leaf(l, f)
        def set_trainable(l, b):
            apply_leaf(l, lambda m: set_trainable_attr(m, b))

        # You can unfreeze the last layer of bert by calling set_trainable(model.bert.encoder.layer[23], True)
        set_trainable(self.bert, False)
        for i in range(start_layer, end_layer+1):
            set_trainable(self.bert.encoder.layer[i], True)