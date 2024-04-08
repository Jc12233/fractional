delta_f = 312.5*1000;
T = 1e-7;
M = 32;
N = 32;
D_dd = ones(M,N);
F_M = dftmtx(M);
F_N = dftmtx(N);
F_N_conj_trans = F_N';
% OTFS transmitter
D_tf = F_M*D_dd*F_N_conj_trans;
d_dd = D_dd(:);
G = ones(M,M);
s = kron(F_N_conj_trans, G)*d_dd;

% Channel
K = 3;  % 信道数
delay = rand(K,1);    % 时延
dop = rand(K,1);  % doppler频移
l_tao =[1 2 3] ;
k_dop = [1 2 3]  ;
h = rand(K,1)+1i*rand(K,1); % 信道增益
z = exp(1i*2*pi/M/N);
H = zeros(M*N,M*N);
H_n = zeros(M*N,M*N);
cyclic_matrix = eye(M*N);
% 行向下移一位变为循环矩阵
delay_matrix = circshift(cyclic_matrix,1,1);
doppler_vector = z.^(0:M*N-1);
doppler_matrix = diag(doppler_vector);
for j = 1:K
    temp = h(j)*(delay_matrix^(floor(dop(j)*M*delta_f)))*(doppler_matrix^(floor(delay(j)*N*T)));
    H = H+temp;
end
F_MN = dftmtx(M*N);
F_MN_conj_trans = F_MN';
TAO = conj(doppler_matrix);
for j = 1:K
    temp = h(j)*F_MN_conj_trans*(TAO^(floor(dop(j)*M*delta_f)))*F_MN*(doppler_matrix^(floor(delay(j)*N*T)));
    H_n = H_n+temp;
end

% receiver
sigma = 1;
v = sigma*(randn(M*N,1)+1i*randn(M*N,1));
r = H*s+v;
r_n = H_n*s+v;
phi = zeros(M*N,M*N);
% 时延与频移字典
for m = 1:M
    for n = 1:N
        phi(:,(m-1)*N+n) = F_MN_conj_trans*(TAO^m)*F_MN*(doppler_matrix^n)*kron(F_N_conj_trans,G)*d_dd;
    end
end
% 信道字典
h_dic = zeros(M*N,1);
h_dic(l_tao*N+k_dop+1) = h;
r_dic = phi*h+v; % 应与r相同

% OMPFR
N_ite = 50;     % 算法迭代次数
R = M;          % phi矩阵截断行数
tol = 1/sqrt(M*N*sigma^2);      % 算法结束容差
phi_hat = phi(1:R,:);
r_hat = r_n(1:R,:);
h_ompfr= OMPFR(r_n,r_hat,phi_hat,N_ite,tol,M,N,d_dd,G);


