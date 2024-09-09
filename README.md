#求解四种完全配置的典型PDEs的实验代码
Navier-Stokes 方程:
■(∂u/∂t+u ∂u/∂x+v ∂u/∂y&=-∂p/∂x+ν((∂^2 u)/(∂x^2 )+(∂^2 u)/(∂y^2 ))+F_x@∂v/∂t+u ∂v/∂x+v ∂v/∂y&=-∂p/∂y+ν((∂^2 v)/(∂x^2 )+(∂^2 v)/(∂y^2 ))+F_y )
反应-扩散方程 (RD):
■(∂u/∂t&=μ((∂^2 u)/(∂x^2 )+(∂^2 u)/(∂y^2 ))+u-u^3-v+0.01@∂v/∂t&=μ((∂^2 v)/(∂x^2 )+(∂^2 v)/(∂y^2 ))+0.25(u-v) )
黏性Burgers方程:
■(∂u/∂t+u ∂u/∂x+v ∂u/∂y&=μ((∂^2 u)/(∂x^2 )+(∂^2 u)/(∂y^2 ))+(1-u^2-v^2 )u+(u^2+v^2 )v@∂v/∂t+u ∂v/∂x+v ∂v/∂y&=μ((∂^2 v)/(∂x^2 )+(∂^2 v)/(∂y^2 ))-(u^2+v^2 )u+(1-u^2-v^2 )v)
泊松方程(PE):
对于 u 的方程：∂u/∂t=-u ∂u/∂x-v ∂u/∂y+μ((∂^2 u)/(∂x^2 )+(∂^2 u)/(∂y^2 ))+(1-u^2-v^2)u
对于 v 的方程：∂v/∂t=-u ∂v/∂x-v ∂v/∂y+μ((∂^2 v)/(∂x^2 )+(∂^2 v)/(∂y^2 ))-(u^2+v^2)v
分别编写代码求解。

#安装依赖
pip install -r requirements.txt

#代码文件说明
KAN网络求解黏性burgers方程:[KAN]burgers.py
KAN网络求解NS方程:[KAN]NS.py
KAN网络求解泊松方程PE:[KAN]PE.py
KAN网络求解扩散方程RD方程:[KAN]RD.py
MLP网络求解黏性burgers方程:[MLP]burgers.py
MLP网络求解NS方程:[MLP]NS.py
MLP网络求解扩散方程RD方程:[MLP]RD.py
MLP网络求解泊松方程PE:[MLP]PE.py

#数值解法
Kelvin-Helmholtz 不稳定性NS方程求解
使用数据集KFvorticity_Re100_N50_T500.npy,但是没有取得理想的实验结果
NS方程数值解法
使用数据集NavierStokes_V1e-5_N1200_T20.mat,但是没有取得理想的实验结果

#数据集
https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-


#本程序参考了KAN的网络结构 
@article{liu2024kan,
  title={KAN: Kolmogorov-Arnold Networks},
  author={Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and Ruehle, Fabian and Halverson, James and Solja{\v{c}}i{\'c}, Marin and Hou, Thomas Y and Tegmark, Max},
  journal={arXiv preprint arXiv:2404.19756},
  year={2024}
  
