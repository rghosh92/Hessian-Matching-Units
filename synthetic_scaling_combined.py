import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from skdim.datasets import BenchmarkManifolds
from hmu_mods import *
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random
import gc
import pandas as pd

torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')
torch.backends.cudnn.deterministic = True

random.seed(11)
np.random.seed(11)
torch.manual_seed(11)
torch.cuda.manual_seed(0)

gc.collect()
torch.cuda.empty_cache()

def train_and_eval(model, train_data, test_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_loader = DataLoader(TensorDataset(*train_data), batch_size=200,
                              shuffle=True, generator = torch.Generator(device=device))
    test_loader = DataLoader(TensorDataset(*test_data), batch_size=400
                             , generator = torch.Generator(device=device))
    
    for epoch in range(500):
        model.train()
        total_loss = 0
        total_samples = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)  # scale by batch size
            total_samples += batch_x.size(0)
        
    avg_train_loss = total_loss / total_samples
    
    print('Average Train loss:', avg_train_loss)
            
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits, _ = model(batch_x)
            preds = logits.argmax(1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    
    # print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {correct/total:.4f}")
        
    return correct / total
class ManifoldRegistry:
    # Maps user names to (native_d, native_D, epsilon_default)
    # If native_d is None, it means the manifold supports 'any' d natively.
    CONFIG = {
        "Mp1_Paraboloid": {"native_d": None, "dim_func": lambda d: 3*(d+1), "eps": 0.3},
        "M13a_Scurve":    {"native_d": 2,    "dim_func": lambda d: 3,      "eps": 0.1},
        "M1_Sphere":      {"native_d": None, "dim_func": lambda d: d+1,    "eps": 0.3},
        "M11_Moebius":    {"native_d": 2,    "dim_func": lambda d: 3,      "eps": 0.3},
        "M7_Roll":        {"native_d": 2,    "dim_func": lambda d: 3,      "eps": 0.3},
        "M4_Nonlinear":   {"native_d": None, "dim_func": lambda d: 2*d,    "eps": 0.3}
    }


class BinaryManifoldFactory:
    def __init__(self, manifold_gen):
        self.generator = manifold_gen

    def _get_local_normals(self, X, d_intrinsic, n_neighbors=15):
        """
        Estimates a normal vector for each point using Local PCA.
        The normal is the (d+1)-th principal component.
        """
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
        distances, indices = nn.kneighbors(X)
        
        normals = np.zeros_like(X)
        for i in range(len(X)):
            neighbors = X[indices[i]]
            # Local PCA
            pca = PCA(n_components=d_intrinsic + 1)
            pca.fit(neighbors)
            # The last component is the direction with the least variance (the normal)
            normals[i] = pca.components_[-1]
            
        # Ensure consistent orientation (simple heuristic)
        return normals
    def generate_binary_data(self, name, n_samples, d_target, D_target, epsilon=0.1):
        X_ambient = None
        conf = ManifoldRegistry.CONFIG.get(name)
        epsilon = conf['eps']

        # --- STEP 1: Determine Native vs Fallback ---
        if conf['native_d'] is None:
            # Manifold supports variable d (like Sphere or Paraboloid)
            X_core = self.generator.generate(name=name, n=n_samples, d=d_target, dim=conf['dim_func'](d_target))
        else:
            # Manifold has a fixed ID (like S-Curve d=2)
            if d_target == conf['native_d']:
                X_core = self.generator.generate(name=name, n=n_samples, d=conf['native_d'], dim=conf['dim_func'](conf['native_d']))
            elif d_target > conf['native_d']:
                # FALLBACK: Stack the fixed-ID manifold
                repeats = int(np.ceil(d_target / conf['native_d']))
                chunks = [self.generator.generate(name=name, n=n_samples, d=conf['native_d'], dim=conf['dim_func'](conf['native_d'])) for _ in range(repeats)]
                X_core = np.concatenate(chunks, axis=1)
            else:
                return None, None # Cannot down-sample ID
                
        # --- STEP 2: Project to AMB_RANGE (100) ---
        curr_D = X_core.shape[1]
        # print(D_target)
        if curr_D < D_target:
            # Orthogonal Isometry to boost ambient dim to 100
            proj = np.random.randn(D_target, curr_D)
            q, _ = np.linalg.qr(proj)
            X_ambient = X_core @ q.T
        elif curr_D == D_target:
            X_ambient = X_core
        else:
            return None, None # Native D is already larger than 100

        # --- STEP 3: Normalization & Fixed Shift Perturbation ---
        # Normalize
        # x_min, x_max = X_ambient.min(axis=0), X_ambient.max(axis=0)
        # X_ambient = (X_ambient - x_min) / np.where((x_max - x_min) == 0, 1, x_max - x_min)
        
        # Shift
        proj = np.random.randn(D_target, D_target)
        
        # 2. QR decomposition gives us orthonormal columns
        # q shape will be (D_target, curr_D)
        q, _ = np.linalg.qr(proj)
        
        # 3. Project: (N x curr_D) @ (curr_D x D_target)
        # This is a rigid rotation/embedding into higher space.
        # Distances are preserved: ||X_core_a - X_core_b|| == ||X_amb_a - X_amb_b||
        X_ambient = X_ambient @ q.T
        
        normals = self._get_local_normals(X_ambient, d_intrinsic=d_target)
        
        half_n = n_samples // 2
        
        
        X_1 = X_ambient[half_n:]
        
        X_0 = X_ambient[:half_n] + epsilon * normals[:half_n]
     

        X = np.vstack([X_0, X_1])
        y = np.concatenate([np.zeros(half_n), np.ones(half_n)])
        
        
        indices = np.random.permutation(len(X))
        return torch.tensor(X[indices], dtype=torch.float32), torch.tensor(y[indices], dtype=torch.long)

      
   
    
def get_trainable_params(model):
    """Calculates the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


    
# Configuration
ID_RANGE = [2,4,6,8] # Varying d
AMB_RANGE = [20,50,100]
MADDEN_RANGE = [128,64,32,4] # Hidden layer width
MANIFOLDS_TO_TEST = ["Mp1_Paraboloid","M1_Sphere","M4_Nonlinear", "M11_Moebius","M13a_Scurve", "M7_Roll" ]
epsilons = [0.1,0.1,0.5,0.5,0.1,0.5]

TRAIN_N = 10000

TEST_N = 10000
classes = 2

factory = BinaryManifoldFactory(BenchmarkManifolds(random_state=42))


all_results = [] 


for d in ID_RANGE:
    for D in AMB_RANGE:
        for temp in range(len(MANIFOLDS_TO_TEST)):
        
            m_name =  MANIFOLDS_TO_TEST[temp]
            
            
    
            # Generate Data
            
            X, y = factory.generate_binary_data(m_name, n_samples=TRAIN_N+TEST_N,
                                                d_target=d, D_target=D,epsilon = epsilons[temp])
            
            if X is None:
                print(f"Skipping incompatible config: {m_name} (d={d}, D={D})")
                continue
            
            print(f"Config: {m_name} (d={d}, D={D})")
            print(X.shape)
            print(y.shape)

            train_x, train_y = X[:TRAIN_N], y[:TRAIN_N]
            test_x, test_y = X[TRAIN_N:], y[TRAIN_N:]
            
            for madden in MADDEN_RANGE:
                print(madden)
                configs = [
                    ("SimpleMLP",  SimpleMLP_synthetic(D, num_classes=classes, madden = 2*madden)),    
                    
                    ("SimpleHMU-8", SimpleHMUMLP_synthetic(D, num_slices=8, 
                                                      num_classes=classes, madden = int(madden/4), relu=False)),
                    ("SimpleRBF", SimpleHMUMLP_synthetic(D, num_slices=0, 
                                                      num_classes=classes, madden = int(2*madden), relu=False))
                    
                    
                ]
                for model_name, net in configs:
                    # Note: We re-initialize the net for the current FIXED_AMB_DIM
                    print(f"Training {model_name}...")
                    acc = train_and_eval(net, (train_x, train_y), (test_x, test_y))
                    total_params = get_trainable_params(net)
                    
                    all_results.append({
                        "m_name": m_name,
                        "id": d,
                        "amb_dim": D,
                        "madden": madden,
                        "model": model_name,
                        "accuracy": acc,
                        "trainable_params": total_params,  
                        "compression_ratio": d / D 
                    })
                    print(all_results[-1])
            
            df = pd.DataFrame(all_results)
            df.to_csv("synthetic_scaling_sweep_all.csv", index=False)

            # df.to_csv("synthetic_scaling_sweep_id4_id6_id8_amb20_50_100_M3.csv", index=False)
