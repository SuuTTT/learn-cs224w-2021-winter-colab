import torch
import torch.nn.functional as F
print(torch.__version__)

# The PyG built-in GCNConv
from torch_geometric.nn import GCNConv

import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

#dataset
dataset_name = 'ogbn-arxiv'
dataset = PygNodePropPredDataset(name=dataset_name,
                                 transform=T.ToSparseTensor())

data = dataset[0]
print(data)

data.adj_t=data.adj_t.to_symmetric()
split_idx=dataset.get_idx_split()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_idx=split_idx['train'].to(device)

#models <-Construct the network as showing in the figure
class GCN(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim,output_dim,num_layers,dropout):
    super(GCN, self).__init__()
    self.convs=None
    self.bns=None
    self.dropout=dropout
    self.num_layers=num_layers
    self.convs=torch.nn.ModuleList()
    self.convs.append(GCNConv(input_dim, hidden_dim))
    for i in range(num_layers-2):
      self.convs.append(GCNConv(hidden_dim,hidden_dim))
    self.convs.append(GCNConv(hidden_dim,output_dim))
    self.bns=torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for i in range(num_layers-1)])
    self.softmax=torch.nn.LogSoftmax()
  
  def reset_parameters(self):
    for conv in self.convs:
      conv.reset_parameters()
    for bn in self.bns:
      bn.reset_parameters()
  def forward(self,x, adj_t):
    for i in range(self.num_layers-1):
      x=self.convs[i](x, adj_t)
      x=self.bns[i](x)
      x=F.ReLU(x)
      x=F.dropout(x,self.dropout,self.training)
    x=self.conv[-1](x,adj_t)
    x=self.softmax(x)
    return x

def train(model, data, train_idx, optimizer, loss_fn):
    # TODO: Implement this function that trains the model by 
    # using the given optimizer and loss_fn.
    model.train()
    loss = 0

    ############# Your code here ############
    ## Note:
    ## 1. Zero grad the optimizer
    ## 2. Feed the data into the model
    ## 3. Slicing the model output and label by train_idx
    ## 4. Feed the sliced output and label to loss_fn
    ## (~4 lines of code)
    optimizer.zero_grad()
    x=data.x[train_idx]
    y=data.y[train_idx,0]
    l=loss_fn(model(x,data.adj_t),y)
    l.backward()
    optimizer.step()
    return l.item()

    # Test function here
@torch.no_grad()
def test(model, data, split_idx, evaluator):
    # TODO: Implement this function that tests the model by 
    # using the given split_idx and evaluator.
    model.eval()
    #Sets the module in evaluation mode.

    # The output of model on all data
    out = None

    ############# Your code here ############
    ## (~1 line of code)
    ## Note:
    ## 1. No index slicing here
    out=model(data.x,data.adj_t)
    #########################################
    y_pred=torch.argmax(out,dim=-1,keep_dims=True)
    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    return train_acc, valid_acc, test_acc
  
  #hyperparameters
args = {
    'device': device,
    'num_layers': 3,
    'hidden_dim': 256,
    'dropout': 0.5,
    'lr': 0.01,
    'epochs': 10,
}
#instanize
model = GCN(data.num_features, args['hidden_dim'],
            dataset.num_classes, args['num_layers'],
            args['dropout']).to(device)
evaluator = Evaluator(name='ogbn-arxiv')


#train
import copy
# reset the parameters to initial random value
model.reset_parameters()
optimizer=torch.optim.Adam(model.parameters(),args['lr'])
loss_fn=F.nll_loss
print('begin trainging')
best_acc=0
best_model=None
for epoch in range(args['epochs']):
  loss=train(model,data,train_idx, optimizer, loss_fn)
  train_acc, valid_acc, test_acc=test(model,data,split_idx,evaluator)
  if valid_acc>best_acc:
    best_acc=valid_acc
    best_model=copy.deepcopy(model)
  print(f'epoch:{epoch:02d},',f'train_acc:{train_acc:02d},',f'valid_acc:{valid_acc:02d}',f'test_acc:{test_acc:02}')