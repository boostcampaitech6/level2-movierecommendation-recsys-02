import numpy as np
import torch
import torch.nn as nn

# FM
class FMLayer(nn.Module):
    def __init__(self, input_dim, factor_dim):
        '''
        Parameter
            input_dim: Input dimension in sparse representation (2652 in MovieLens-100k)
            factor_dim: Factorization dimension
        '''
        super(FMLayer, self).__init__()
        self.v = nn.Parameter(
            torch.empty(input_dim, factor_dim)  # FILL HERE : Fill in the places `None` #
            , requires_grad = True
        )

    def square(self, x):
        return torch.pow(x,2)

    def forward(self, x):
        '''
        Parameter
            x: Float tensor of size "(batch_size, input_dim)"
        '''
        square_of_sum =  self.square(torch.sum(torch.matmul(x, self.v), dim=1)) # FILL HERE : Use `torch.matmul()` and `self.square()` #
        sum_of_square =  torch.sum(torch.matmul(self.square(x), self.square(self.v)), dim=1) # FILL HERE : Use `torch.matmul()` and `self.square()` #

        return 0.5 * torch.sum(square_of_sum - sum_of_square, dim=0)

class FactorizationMachine(nn.Module):
    def __init__(self, input_dim, factor_dim):
        '''
        Parameter
            input_dim: Input dimension in sparse representation (2652 in MovieLens-100k)
            factor_dim: Factorization dimension
        '''
        super(FactorizationMachine, self).__init__()

        self.linear = nn.Linear(input_dim, 1, bias=True) # FILL HERE : Fill in the places `None` #
        self.fm = FMLayer(input_dim, factor_dim) # FILL HERE : Fill in the places `None` #

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, FMLayer):
                nn.init.normal_(m.v, 0, 0.01)


    def forward(self, x):
        '''
        Parameter
            x: Long tensor of size "(batch_size, input_dim)"

        Return
            y: Float tensor of size "(batch_size)"
        '''
        x = x.float()
        y = self.linear(x).squeeze(1) + self.fm(x) # FILL HERE : Use `self.linear()` and `self.fm()` #
        y = torch.sigmoid(y)
        return y
    
# FFM
class FeaturesLinear(nn.Module):

    def __init__(self, field_dims: np.ndarray, output_dim: int=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        
        # self.linear = nn.Linear(input_dim, 1, bias=True)
        #linear(x)
    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)`` = (256,9)
        :return : (batch_size, output_dim=1)
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0) #[256,9]
        return torch.sum(self.fc(x), dim=1) + self.bias #self.fc(x) = (256,9,1) #self.bias = [0.] => [256,1]
    
class FieldAwareFactorizationMachine(nn.Module):

    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)`` #[256,9]
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        
        return ix #[256,36,8]
    
class FieldAwareFactorizationMachineModel(nn.Module):

    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.ffm = FieldAwareFactorizationMachine(field_dims, embed_dim)
        
    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        ffm_term = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        x = self.linear(x) + ffm_term
        return torch.sigmoid(x.squeeze(1))
    
# DeepFM    
class DeepFM(nn.Module):

    def __init__(self, input_dims, embedding_dim, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()
        total_input_dim = int(sum(input_dims)) # 입력 특성의 차원 n_user + n_movie + n_genre

        # Fm component의 constant bias term과 1차 bias term
        self.bias = nn.Parameter(torch.zeros((1,)))
        self.fc = nn.Embedding(total_input_dim, 1)

        self.embedding = nn.Embedding(total_input_dim, embedding_dim)
        self.embedding_dim = len(input_dims) * embedding_dim


        mlp_layers = [] # mlp hidden layer
        for i, dim in enumerate(mlp_dims):
            if i==0:
                mlp_layers.append(nn.Linear(self.embedding_dim, dim))
            else:
                mlp_layers.append(nn.Linear(mlp_dims[i-1], dim))
            mlp_layers.append(nn.ReLU(True))
            mlp_layers.append(nn.Dropout(drop_rate))
        mlp_layers.append(nn.Linear(mlp_dims[-1], 1))
        self.mlp_layers = nn.Sequential(*mlp_layers)

    def fm(self, x):
        # x : (batch_size, total_num_input)
        embed_x = self.embedding(x)

        fm_y = self.bias + torch.sum(self.fc(x), dim=1)

        square_of_sum = torch.sum(embed_x, dim=1) ** 2
        sum_of_square = torch.sum(embed_x ** 2, dim=1)
        fm_y += 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        return fm_y

    def mlp(self, x):
        embed_x = self.embedding(x)

        inputs = embed_x.view(-1, self.embedding_dim)
        mlp_y = self.mlp_layers(inputs)
        return mlp_y

    def forward(self, x):
        embed_x = self.embedding(x)

        # fm component
        fm_y = self.fm(x).squeeze(1)

        # deep component
        mlp_y = self.mlp(x).squeeze(1)

        # 시그모이드 함수를 사용하여 ctr 1/0 에 대한 확률 값을 출력
        y = torch.sigmoid(fm_y + mlp_y)
        return y