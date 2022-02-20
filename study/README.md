# todo 

- [ ] 0 graph data 
- [ ] 1 node embedding
- [ ] 2 gnn
- [ ] 3 gat
- [ ] 4 knowledge graph

# colab0

心得 

系统学的话用文档，安装、教程、细节都有。小问题直接面向csdn。



## networkx

nx包 基类graph类用三重嵌套dict。

最内层用来维护node feature。 中层是邻接链表。外层是node集合。

用了工厂模式，更新时使用dict的update方法



## pytorch metric

pytorch-geometric包 

windows上	安装 pip install 的话很慢， conda好很多，最后在colab上跑

# colab1

最简单的训练



## pytorch

### 自动求导

初始化时要requresgrad`x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)`，重新算时清空`x.grad.zero_()` 对应` optimizer.zero_grad()`

backward的对象一般是关于参数的函数（标量）`y = 2 * torch.dot(x, x)`。参数要作为整体tensor写，并且只能用函数， 注释中用下标取回使得参数无法更新（猜测取地址不是计算？函数赋值一边才是）。

```python
#y_hat=[torch.dot(emb.weight.data[u],emb.weight.data[v]) for u,v in train_edge]
    embs=emb(train_edge)#[2,312,16]
    #print(embs.shape)
    #y_hat=torch.dot(embs[0],embs[1])#[312]
    y_hat=embs[0]*embs[1]
    y_hat=torch.sum(y_hat,dim=1)
    #y_hat=torch.tensor(y_hat,requires_grad=True)
    y_hat=torch.sigmoid(y_hat)
    #print(y_hat.shape)
    l=loss_fn(y_hat,train_label)
    l.backward()
    optimizer.step()
```

### embedding

 emb=nn.Embedding(num_node,embedding_dim)

 emb.weight.data=torch.rand(num_node,embedding_dim)



如果y不是标量，需要带参数，不常见:

```
import torch

x=torch.randn(3)
x=torch.autograd.Variable(x,requires_grad=True)#生成变量
print(x)#输出x的值
y=x*2
y.backward(gradient=torch.FloatTensor([1,0.1,0.01]))#自动求导
print(x.grad)#求对x的梯度

```

## sklearn pca+ plt

```
def visualize_emb(emb):
  X = emb.weight.data.numpy()
  pca = PCA(n_components=2)
  components = pca.fit_transform(X)
  plt.figure(figsize=(6, 6))
  club1_x = []
  club1_y = []
  club2_x = []
  club2_y = []
  for node in G.nodes(data=True):
    if node[1]['club'] == 'Mr. Hi':  #node的形式：第一个元素是索引，第二个元素是attributes字典
      club1_x.append(components[node[0]][0])
      club1_y.append(components[node[0]][1])
    else:
      club2_x.append(components[node[0]][0])
      club2_y.append(components[node[0]][1])
  plt.scatter(club1_x, club1_y, color="red", label="Mr. Hi")
  plt.scatter(club2_x, club2_y, color="blue", label="Officer")
  plt.legend()
  plt.show()
```



## random-negative sample

```
 non_edges_one_side=list(enumerate(nx.non_edges(G)))
  neg_edge_list_indices=random.sample(range(0,len(non_edges_one_side)),num_neg_samples)  #取样num_neg_samples长度的索引
  for i in neg_edge_list_indices:
    neg_edge_list.append(non_edges_one_side[i][1])
```

# colab2

用GNN做两个任务，节点分类和图分类。

## 模型：

节点预测是conv - batch norm - relu - dropout ... - conv - Logsoftmax

 图预测是节点预测之后加一个 global_mean_pool - Linear 来二分类

```python
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):
        # TODO: Implement this function that initializes self.convs, 
        # self.bns, and self.softmax.

        super(GCN, self).__init__()

        # A list of GCNConv layers
        self.convs = None

        # A list of 1D batch normalization layers
        self.bns = None

        # The log softmax layer
        self.softmax = None

        ############# Your code here ############
        ## Note:
        ## 1. You should use torch.nn.ModuleList for self.convs and self.bns
        ## 2. self.convs has num_layers GCNConv layers
        ## 3. self.bns has num_layers - 1 BatchNorm1d layers
        ## 4. You should use torch.nn.LogSoftmax for self.softmax
        ## 5. The parameters you can set for GCNConv include 'in_channels' and 
        ## 'out_channels'. More information please refer to the documentation:
        ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
        ## 6. The only parameter you need to set for BatchNorm1d is 'num_features'
        ## More information please refer to the documentation: 
        ## https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        ## (~10 lines of code)
        self.convs=torch.nn.ModuleList([GCNConv(input_dim,hidden_dim)])
        self.convs.extend(torch.nn.ModuleList([GCNConv(hidden_dim,hidden_dim) for i in range(num_layers-2)]))
        self.convs.append(GCNConv(hidden_dim,output_dim))
        self.bns=torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for i in range(num_layers-1)])
        self.softmax=torch.nn.LogSoftmax()
        print(self.convs)
        #########################################

        # Probability of an element to be zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds
        self.num_layers=num_layers# why no num_layers, because not instanize

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        # TODO: Implement this function that takes the feature tensor x,
        # edge_index tensor adj_t and returns the output tensor as
        # shown in the figure.

        out = None

        ############# Your code here ############
        ## Note:
        ## 1. Construct the network as showing in the figure
        ## 2. torch.nn.functional.relu and torch.nn.functional.dropout are useful
        ## More information please refer to the documentation:
        ## https://pytorch.org/docs/stable/nn.functional.html
        ## 3. Don't forget to set F.dropout training to self.training
        ## 4. If return_embeds is True, then skip the last softmax layer
        ## (~7 lines of code)
        for i in range(len(self.convs)-1):
          x=self.convs[i](x,adj_t)
          x=self.bns[i](x)
          x=F.relu(x)
          x=F.dropout(x,self.dropout,self.training)
        x=self.convs[-1](x,adj_t)
        if not self.return_embeds:
          x=self.softmax(x)
          
        #########################################
        out=x
        return out
print("run")
```



## 数据集划分

都是使用get_idx_split来随机划分train, valid, test

作业中，节点预测只对一个图训练（数据集有多个图）

图预测将32个图作为一个batch来训练。

```
dataset_name = 'ogbn-arxiv'
dataset = PygNodePropPredDataset(name=dataset_name,
                                 transform=T.ToSparseTensor())
data = dataset[0]

# Make the adjacency matrix to symmetric
data.adj_t = data.adj_t.to_symmetric()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# If you use GPU, the device should be cuda
print('Device: {}'.format(device))

data = data.to(device)
split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)
type(split_idx)
print(data)
----图---
# Load the data sets into dataloader
# We will train the graph classification task on a batch of 32 graphs
# Shuffle the order of graphs for training set
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True, num_workers=0)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False, num_workers=0)
```

## 训练

```python
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
    y_hat=model(data.x,data.adj_t)[train_idx]
    y=data.y[train_idx,0] #shape=[...]
    print(y_hat.shape,y.shape,y[0].shape)

    #y=y[0]# shape=...
    loss=loss_fn(y_hat,y)

    #########################################

    loss.backward()
    optimizer.step()

    return loss.item()
```

```
#batch 训练
def train(model, device, data_loader, optimizer, loss_fn):
    # TODO: Implement this function that trains the model by 
    # using the given optimizer and loss_fn.
    model.train()
    loss = 0

    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
      batch = batch.to(device)

      if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
          pass
      else:
        ## ignore nan targets (unlabeled) when computing training loss.
        is_labeled = batch.y == batch.y

        ############# Your code here ############
        ## Note:
        ## 1. Zero grad the optimizer
        ## 2. Feed the data into the model
        ## 3. Use `is_labeled` mask to filter output and labels
        ## 4. You might change the type of label
        ## 5. Feed the output and label to loss_fn
        ## (~3 lines of code)
        optimizer.zero_grad()
        #这里传batch.x报错
        y_hat=model(batch)
        #不float报错
        y=batch.y[is_labeled].float()
        loss=loss_fn(y_hat[is_labeled],y)


        #########################################

        loss.backward()
        optimizer.step()

    return loss.item()
```





## test

测试直接调用数据集自带的函数

```
# Test function here
@torch.no_grad()
def test(model, data, split_idx, evaluator):
    # TODO: Implement this function that tests the model by 
    # using the given split_idx and evaluator.
    model.eval()

    # The output of model on all data
    out = None

    ############# Your code here ############
    ## (~1 line of code)
    ## Note:
    ## 1. No index slicing here
    out=model(data.x,data.adj_t)
    #########################################

    y_pred = out.argmax(dim=-1, keepdim=True)

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
 
 model = GCN(data.num_features, args['hidden_dim'],
            dataset.num_classes, args['num_layers'],
            args['dropout']).to(device)
evaluator = Evaluator(name='ogbn-arxiv')
```

# colab3

从0实现GnnConvs类，graphSAGE, GAT。





## GraphSage Implementation

代码的框架设计 分成3个 ***\*forward\****, ***\*message\**** and ***\*aggregate\**** functions。公式中 Wl, Wr 应该只是左右的含义，共享。

forward调用propagate ,详见message pass子类的设计[Creating Message Passing Networks — pytorch_geometric 2.0.4 documentation (pytorch-geometric.readthedocs.io)](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html)。

aggregate调用scatter函数。[Scatter — pytorch_scatter 2.0.9 documentation (pytorch-scatter.readthedocs.io)](https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html) 一般用agg指定

```

source=torch.tensor([1,1,2,2,3,3])
source=source.reshape((1,3,2))
index=torch.tensor([0,0,1])
out=torch.zeros((1,2,2))
torch_scatter.scatter(source,index,1,reduce="sum")
```







Now let's start working on our own implementation of layers! This part is to get you familiar with how to implement Pytorch layer based on Message Passing. You will be implementing the ***\*forward\****, ***\*message\**** and ***\*aggregate\**** functions.



Generally, the ***\*forward\**** function is where the actual message passing is conducted. All logic in each iteration happens in ***\*forward\****, where we'll call ***\*propagate\**** function to propagate information from neighbor nodes to central nodes.  So the general paradigm will be pre-processing -> propagate -> post-processing.



Recall the process of message passing we introduced in homework 1. ***\*propagate\**** further calls ***\*message\**** which transforms information of neighbor nodes into messages, ***\*aggregate\**** which aggregates all messages from neighbor nodes into one, and ***\*update\**** which further generates the embedding for nodes in the next iteration.



Our implementation is slightly variant from this, where we'll not explicitly implement ***\*update\****, but put the logic for updating nodes in ***\*forward\**** function. To be more specific, after information is propagated, we can further conduct some operations on the output of ***\*propagate\****. The output of ***\*forward\**** is exactly the embeddings after the current iteration.



In addition, tensors passed to ***\*propagate()\**** can be mapped to the respective nodes $i$ and $j$ by appending _i or _j to the variable name, .e.g. x_i and x_j. Note that we generally refer to $i$ as the central nodes that aggregates information, and refer to $j$ as the neighboring nodes, since this is the most common notation.



Please find more details in the comments. One thing to note is that we're adding ***\*skip connections\**** to our GraphSage. Formally, the update rule for our model is described as below:
$$
\begin{equation}

h_v^{(l)} = W_l\cdot h_v^{(l-1)} + W_r \cdot AGG(\{h_u^{(l-1)}, \forall u \in N(v) \})

\end{equation}
$$




For simplicity, we use mean aggregations where:
$$
\begin{equation}

AGG(\{h_u^{(l-1)}, \forall u \in N(v) \}) = \frac{1}{|N(v)|} \sum_{u\in N(v)} h_u^{(l-1)}

\end{equation}
$$






Additionally, $\ell$-2 normalization is applied after each iteration.

```python
class GraphSage(MessagePassing):
    
    def __init__(self, in_channels, out_channels, normalize = True,
                 bias = False, **kwargs):  
        super(GraphSage, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_l = None
        self.lin_r = None

        ############################################################################
        # TODO: Your code here! 
        # Define the layers needed for the message and update functions below.
        # self.lin_l is the linear transformation that you apply to embedding 
        #            for central node.
        # self.lin_r is the linear transformation that you apply to aggregated 
        #            message from neighbors.
        # Our implementation is ~2 lines, but don't worry if you deviate from this.
        self.lin_l=Linear(in_channels,out_channels)  #Wl
        self.lin_r=Linear(in_channels,out_channels)  #Wr

        ############################################################################

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size = None):
        """"""

        out = None

        ############################################################################
        # TODO: Your code here! 
        # Implement message passing, as well as any post-processing (our update rule).
        # 1. First call propagate function to conduct the message passing.
        #    1.1 See there for more information: 
        #        https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
        #    1.2 We use the same representations for central (x_central) and 
        #        neighbor (x_neighbor) nodes, which means you'll pass x=(x, x) 
        #        to propagate.
        # 2. Update our node embedding with skip connection.
        # 3. If normalize is set, do L-2 normalization (defined in 
        #    torch.nn.functional)
        # Our implementation is ~5 lines, but don't worry if you deviate from this.
        #out=self.propagate(edge_index,x=(x,x),size=size)
        #20210708尝试一下直接传x。就确实我也没搞懂为什么要传(x,x)，我寻思了一下应该是和直接传x一样的
        out=self.propagate(edge_index,x=x,size=size)
        x=self.lin_l(x)
        out=self.lin_r(out)
        out=out+x#skip connection
        if self.normalize:
            out=F.normalize(out)

        ############################################################################

        return out

    def message(self, x_j):

        out = None

        ############################################################################
        # TODO: Your code here! 
        # Implement your message function here.
        # Our implementation is ~1 lines, but don't worry if you deviate from this.
        out=x_j

        ############################################################################

        return out

    def aggregate(self, inputs, index, dim_size = None):

        out = None

        # The axis along which to index number of nodes.
        node_dim = self.node_dim  #node_dim的情况可以看PyG那个文档，是MessagePassing的参数：indicates along which axis to propagate.

        ############################################################################
        # TODO: Your code here! 
        # Implement your aggregate function here.
        # See here as how to use torch_scatter.scatter: 
        # https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html#torch_scatter.scatter
        # Our implementation is ~1 lines, but don't worry if you deviate from this.
        out=torch_scatter.scatter(inputs,index,node_dim,dim_size=dim_size,reduce='mean')

        ############################################################################

        return out

```



propagate函数内部的解释

In order to complete the work correctly, we have to understand how the different functions interact with each other. In ***\*propagate\**** we can pass in any parameters we want. For example, we pass in $x$ as an parameter:



... = propagate(..., $x$=($x_{central}$, $x_{neighbor}$), ...)



Here $x_{central}$ and $x_{neighbor}$ represent the features from ***\*central\**** nodes and from ***\*neighbor\**** nodes. If we're using the same representations from central and neighbor, then $x_{central}$ and $x_{neighbor}$ could be identical.



Suppose $x_{central}$ and $x_{neighbor}$ are both of shape N * d, where N is number of nodes, and d is dimension of features.



Then in message function, we can take parameters called $x\_i$ and $x\_j$. Usually $x\_i$ represents "central nodes", and $x\_j$ represents "neighbor nodes". Pay attention to the shape here: $x\_i$ and $x\_j$ are both of shape E * d (***\*not N!\****). $x\_i$ is obtained by concatenating the embeddings of central nodes of all edges through lookups from $x_{central}$ we passed in propagate. Similarly, $x\_j$ is obtained by concatenating the embeddings of neighbor nodes of all edges through lookups from $x_{neighbor}$ we passed in propagate.



Let's look at an example. Suppose we have 4 nodes, so $x_{central}$ and $x_{neighbor}$ are of shape 4 * d. We have two edges (1, 2) and (3, 0). Thus, $x\_i$ is obtained by $[x_{central}[1]^T; x_{central}[3]^T]^T$, and $x\_j$ is obtained by $[x_{neighbor}[2]^T; x_{neighbor}[0]^T]^T$



<font color='red'>For the following questions, DON'T refer to any existing implementations online.</font>

## GAT

核心是注意力系数e，softmax后作为真正计算的系数，最后多头机制进行concat。

e是h_i,h_j的函数a的输出。

Attention mechanisms have become the state-of-the-art in many sequence-based tasks such as machine translation and learning sentence representations. One of the major benefits of attention-based mechanisms is their ability to focus on the most relevant parts of the input to make decisions. In this problem, we will see how attention mechanisms can be used to perform node classification of graph-structured data through the usage of Graph Attention Networks (GATs).



The building block of the Graph Attention Network is the graph attention layer, which is a variant of the aggregation function . Let $N$ be the number of nodes and $F$ be the dimension of the feature vector for each node. The input to each graph attentional layer is a set of node features: $\mathbf{h} = \{\overrightarrow{h_1}, \overrightarrow{h_2}, \dots, \overrightarrow{h_N}$\}, $\overrightarrow{h_i} \in R^F$. The output of each graph attentional layer is a new set of node features, which may have a new dimension $F'$: $\mathbf{h'} = \{\overrightarrow{h_1'}, \overrightarrow{h_2'}, \dots, \overrightarrow{h_N'}\}$, with $\overrightarrow{h_i'} \in \mathbb{R}^{F'}$.



We will now describe this transformation of the input features into higher-level features performed by each graph attention layer. First, a shared linear transformation parametrized by the weight matrix $\mathbf{W} \in \mathbb{R}^{F' \times F}$ is applied to every node. Next, we perform self-attention on the nodes. We use a shared attentional mechanism:
$$
\begin{equation} 

a : \mathbb{R}^{F'} \times \mathbb{R}^{F'} \rightarrow \mathbb{R}.

\end{equation}
$$




This mechanism computes the attention coefficients that capture the importance of node $j$'s features to node $i$:
$$
\begin{equation}

e_{ij} = a(\mathbf{W_l}\overrightarrow{h_i}, \mathbf{W_r} \overrightarrow{h_j})

\end{equation}
$$
The most general formulation of self-attention allows every node to attend to all other nodes which drops all structural information. To utilize graph structure in the attention mechanisms, we can use masked attention. In masked attention, we only compute $e_{ij}$ for nodes $j \in \mathcal{N}_i$ where $\mathcal{N}_i$ is some neighborhood of node $i$ in the graph.



To easily compare coefficients across different nodes, we normalize the coefficients across $j$ using a softmax function:
$$
\begin{equation}

\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}

\end{equation}
$$


For this problem, our attention mechanism $a$ will be a single-layer feedforward neural network parametrized by a weight vector $\overrightarrow{a} \in \mathbb{R}^{F'}$, followed by a LeakyReLU nonlinearity (with negative input slope 0.2). Let $\cdot^T$ represent transposition and $||$ represent concatenation. The coefficients computed by our attention mechanism may be expressed as:


$$
\begin{equation}

\alpha_{ij} = \frac{\exp\Big(\text{LeakyReLU}\Big(\overrightarrow{a_l}^T \mathbf{W_l} \overrightarrow{h_i} + \overrightarrow{a_r}^T\mathbf{W_r}\overrightarrow{h_j}\Big)\Big)}{\sum_{k\in \mathcal{N}_i} \exp\Big(\text{LeakyReLU}\Big(\overrightarrow{a_l}^T \mathbf{W_l} \overrightarrow{h_i} + \overrightarrow{a_r}^T\mathbf{W_r}\overrightarrow{h_k}\Big)\Big)}

\end{equation}
$$


For the following questions, we denote $\alpha_l = [...,\overrightarrow{a_l}^T \mathbf{W_l} \overrightarrow{h_i},...]$ and $\alpha_r = [..., \overrightarrow{a_r}^T \mathbf{W_r} \overrightarrow{h_j}, ...]$.





At every layer of GAT, after the attention coefficients are computed for that layer, the aggregation function can be computed by a weighted sum of neighborhood messages, where weights are specified by $\alpha_{ij}$.



Now, we use the normalized attention coefficients to compute a linear combination of the features corresponding to them. These aggregated features will serve as the final output features for every node.


$$
\begin{equation}

h_i' = \sum_{j \in \mathcal{N}_i} \alpha_{ij} \mathbf{W_r} \overrightarrow{h_j}.

\end{equation}
$$


To stabilize the learning process of self-attention, we use multi-head attention. To do this we use $K$ independent attention mechanisms, or ``heads'' compute output features as in the above equations. Then, we concatenate these output feature representations:


$$
\begin{equation}

  \overrightarrow{h_i}' = ||_{k=1}^K \Big(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^{(k)} \mathbf{W_r}^{(k)} \overrightarrow{h_j}\Big)

\end{equation}
$$


where $||$ is concentation, $\alpha_{ij}^{(k)}$ are the normalized attention coefficients computed by the $k$-th attention mechanism $(a^k)$, and $\mathbf{W}^{(k)}$ is the corresponding input linear transformation's weight matrix. Note that for this setting, $\mathbf{h'} \in \mathbb{R}^{KF'}$.
