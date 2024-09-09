#%%
from kan import *
import torch
import matplotlib.pyplot as plt
from torch import autograd
from tqdm import tqdm

dim = 2
np_i = 21  # 内部点的数量（每个维度）
np_b = 21  # 边界点的数量（每个维度）
ranges = [-1, 1]

model_u = KAN(width=[2,10,2], grid=5, k=3, grid_eps=1.0, noise_scale_base=0.25)
model_p = KAN(width=[2,10,1], grid=5, k=3, grid_eps=1.0, noise_scale_base=0.25)

nu = 0.01  # 黏性系数

def batch_jacobian(func, x, create_graph=False):
    def _func_sum(x):
        return func(x).sum(dim=0)
    return autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1,0,2)

# 定义速度（u, v）和压力（p）的精确解或初始猜测
u_exact = lambda x: torch.sin(torch.pi * x[:, [0]]) * torch.cos(torch.pi * x[:, [1]])
v_exact = lambda x: -torch.cos(torch.pi * x[:, [0]]) * torch.sin(torch.pi * x[:, [1]])
p_exact = lambda x: torch.sin(torch.pi * x[:, [0]]) * torch.sin(torch.pi * x[:, [1]])

# 内部点采样
sampling_mode = 'random'  # 'random' 或 'mesh'
x_mesh = torch.linspace(ranges[0], ranges[1], steps=np_i)
y_mesh = torch.linspace(ranges[0], ranges[1], steps=np_i)
X, Y = torch.meshgrid(x_mesh, y_mesh, indexing="ij")
if sampling_mode == 'mesh':
    x_i = torch.stack([X.reshape(-1,), Y.reshape(-1,)]).permute(1,0)
else:
    x_i = torch.rand((np_i**2, 2)) * 2 - 1

# 边界点采样
helper = lambda X, Y: torch.stack([X.reshape(-1,), Y.reshape(-1,)]).permute(1,0)
xb1 = helper(X[0], Y[0])
xb2 = helper(X[-1], Y[0])
xb3 = helper(X[:,0], Y[:,0])
xb4 = helper(X[:,0], Y[:,-1])
x_b = torch.cat([xb1, xb2, xb3, xb4], dim=0)


steps = 100
alpha = 0.1
log = 1

# 记录损失值
pde_loss_list = []
bc_loss_list = []

def train():
    optimizer_u = LBFGS(model_u.parameters(), lr=1, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
    optimizer_p = LBFGS(model_p.parameters(), lr=1, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

    pbar = tqdm(range(steps), desc='Training')

    for _ in pbar:
        def closure():
            global pde_loss, bc_loss
            optimizer_u.zero_grad()
            optimizer_p.zero_grad()

            u_pred = model_u(x_i)
            p_pred = model_p(x_i)

            u = u_pred[:, [0]]
            v = u_pred[:, [1]]

            # 计算梯度
            u_grad = batch_jacobian(lambda x: model_u(x)[:, [0]], x_i, create_graph=True)
            v_grad = batch_jacobian(lambda x: model_u(x)[:, [1]], x_i, create_graph=True)
            p_grad = batch_jacobian(lambda x: model_p(x)[:, [0]], x_i, create_graph=True)

            u_x, u_y = u_grad[:, 0, 0], u_grad[:, 0, 1]
            v_x, v_y = v_grad[:, 0, 0], v_grad[:, 0, 1]

            p_x, p_y = p_grad[:, 0, 0], p_grad[:, 0, 1]

            # N-S 方程残差
            res_u = u * u_x + v * u_y + p_x - nu * (u_x**2 + u_y**2)
            res_v = u * v_x + v * v_y + p_y - nu * (v_x**2 + v_y**2)

            div = u_x + v_y  # 连续性方程残差

            pde_loss = torch.mean(res_u**2 + res_v**2 + div**2)

            # 边界损失（为简单起见，使用Dirichlet边界条件）
            u_b = model_u(x_b)
            v_b = model_p(x_b)
            u_b_exact = torch.cat([u_exact(x_b), v_exact(x_b)], dim=1)

            bc_loss = torch.mean((u_b - u_b_exact)**2)

            loss = alpha * pde_loss + bc_loss
            loss.backward()
            return loss

        if _ % 5 == 0 and _ < 50:
            model_u.update_grid_from_samples(x_i)
            model_p.update_grid_from_samples(x_i)

        optimizer_u.step(closure)
        optimizer_p.step(closure)

        if _ % log == 0:
            pbar.set_description("pde loss: %.2e | bc loss: %.2e" % (pde_loss.cpu().detach().numpy(), bc_loss.cpu().detach().numpy()))
            # 记录损失
            pde_loss_list.append(pde_loss.cpu().detach().numpy())
            bc_loss_list.append(bc_loss.cpu().detach().numpy())

train()

# 绘制损失图像
plt.figure(figsize=(10, 5))
plt.plot(pde_loss_list, label='PDE Loss')
plt.plot(bc_loss_list, label='BC Loss')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')
plt.show()

# 绘制 pointnet_u 和 pointnet_p 的图像
with torch.no_grad():
    u_pred = model_u(x_i)
    p_pred = model_p(x_i)

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(x_i[:, 0].cpu(), x_i[:, 1].cpu(), c=u_pred[:, 0].cpu(), cmap='jet')
    plt.colorbar()
    plt.title('PointNet_u')
    
    plt.subplot(1, 2, 2)
    plt.scatter(x_i[:, 0].cpu(), x_i[:, 1].cpu(), c=p_pred[:, 0].cpu(), cmap='jet')
    plt.colorbar()
    plt.title('PointNet_p')
    
    plt.show()

#%%
model_p.plot()
#%%
model_u.plot()
#%%
mode = "auto" # "manual"

if mode == "manual":
    # manual mode
    model_p.fix_symbolic(0,0,0,'sin');
    model_p.fix_symbolic(0,1,0,'x^2');
    model_p.fix_symbolic(1,0,0,'exp');
elif mode == "auto":
    # automatic mode
    lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
    model_p.auto_symbolic(lib=lib)
#%%
mode = "auto" # "manual"

if mode == "manual":
    # manual mode
    model_u.fix_symbolic(0,0,0,'sin');
    model_u.fix_symbolic(0,1,0,'x^2');
    model_u.fix_symbolic(1,0,0,'exp');
elif mode == "auto":
    # automatic mode
    lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
    model_u.auto_symbolic(lib=lib)
#%%
formula_p, var_p = model_p.symbolic_formula(floating_digit=5)
formula_u, var_u = model_u.symbolic_formula(floating_digit=5)
formula_u[0]
#%%
