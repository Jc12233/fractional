clear;

% 参数设置
M = 16;                         % 延迟 bins 数量
N = 16;                         % 多普勒 bins 数量
T = 1e-6;                       % 符号持续时间，单位：s
T_guard = 0.1 * T;              % 保护间隔时间，单位：s

K = 1;                          % 目标数量

c = 3e8;                        % 光速，单位：m/s
fc = 24e9;                      % 载波频率，单位：Hz
delta_f = 312.5e3;              % 子载波间距，单位：Hz
B = 10e6;                       % 带宽，单位：Hz
delta_R = c/(2*B);                   % 距离分辨率，单位：m
R_max = 384;                    % 最大不模糊距离，单位：m
delta_V = c*B/(2*M*N*fc);       % 速度分辨率，单位：m/s
V_max = N/2*delta_V;            % 最大速度，单位：m/s
N_ite = 50;                     % 迭代次数
sigma2 = 1;                     % 噪声方差
epsilon = N*M*sigma2;     % 残差阈值

% 物体数据
range_data                      = [33.75, 50.625, 71.25, 97.5];                 % 距离（m）
normalized_delay_data           = range_data/delta_R;           % 归一化延迟
velocity_data                   = 4*[7.63, 38.15, 7.63, -30.52];               % 速度（m/s）
normalized_doppler_shift_data   = N/2+velocity_data/delta_V; % 归一化多普勒频移（Hz）
snr_data                        = [20, 15, 10, 5];                                 % 信噪比（dB）
signal_pow                      = sqrt(10.^(snr_data/10)*sigma2);

% % 将归一化延迟和归一化多普勒频移转换为实际值
% delay = normalized_delay_data * (T / M);
% velocity = normalized_doppler_shift_data * (delta_V / N);

% 生成随机发送数据
rng(1,'twister');
D_dd = 2*randi([0, 1], M, N)-1;


% 使用给定的目标参数
h_k = signal_pow';            % 目标增益
l_tau = normalized_delay_data';               % 整数延迟
k_nu = normalized_doppler_shift_data';                 % 整数多普勒

% 发送 OTFS 信号
G_t = eye(M);               % 发送脉冲塑形矩阵
% G_t = gaussian_pulse_matrix(N,M,T);
G_r = G_t;
s = generate_OTFS_signal(D_dd, M, N, G_t);

% 构造 DD 域雷达信道矩阵
[H_ni, h_ni]= generate_DD_channel(K, h_k, l_tau, k_nu, M, N);
H_n = generate_DD_channel_int(K, h_k, l_tau, k_nu, M, N);

% 生成噪声向量
v = sqrt(sigma2/2) * (randn(M*N, 1) + 1i * randn(M*N, 1));
% v = zeros(M*N,1);

% 接收信号模拟
h = zeros(M*N,1);
h(floor(normalized_delay_data)*N+floor(normalized_doppler_shift_data)+1)=signal_pow;
r_chan = receive_signal_non_integer(H_ni, s, v);
% 接收整形
r = kron(dftmtx(N),G_r)*r_chan;
r_int = receive_signal_integer(H_n, s, v);

% 使用 OMPFR 算法估计目标的位置和速度
R = M*N;  % 截断数量
Phi = kron(dftmtx(N),G_r)*construct_dictionary(M, N, G_t, D_dd);
%Phi = load('data\parameter32.mat').Phi;      % 读取存储数据
Phi_truncated = Phi(1:R, :);  % 对字典进行截断
r_truncated = r(1:R);  % 对接收信号进行截断
h_est_epoint = omp_epoint(r_truncated, Phi_truncated, N_ite, epsilon, M, N, G_r,G_t, D_dd(:));
h_est_fibo = omp_fibo(r_truncated, Phi_truncated, N_ite, epsilon, M, N, G_r,G_t, D_dd(:));
h_est_fast = ompfr_fast(r_truncated, Phi_truncated, N_ite, epsilon, M, N, G_r,G_t, D_dd(:));
% h_int =  omp(r_truncated, Phi_truncated, N_ite, epsilon, M, N);
% h_estimate = ompfr_1(r_truncated, Phi_truncated, N_ite, epsilon, M, N, G_r,G_t, D_dd(:));
% for porjection plot:
% x = [1:M];y = [1:N];[xx,yy] = meshgrid(x,y);surf(xx,yy,reshape(abs(proj_frac),N,M));

% 从估计的 h 中提取位置和速度信息
estimated_range = zeros(K, 1);
estimated_velocity = zeros(K, 1);
for i = 1:K
    % 估计的 h 对应目标的位置和速度
    h_i = h_estimate((i-1)*N+1:i*N);
    % 计算位置和速度
    [~, max_delay_index] = max(abs(h_i));
    [~, max_doppler_index] = max(abs(h_i));
    estimated_range(i) = (max_delay_index - 1) * delta_R;  % 计算估计的范围
    estimated_velocity(i) = (max_doppler_index - 1) * delta_V;  % 计算估计的速度
end


disp('系统仿真完成！');

function Psi = construct_dictionary(M, N, G_t,D_dd)
    % 计算 Gamma 矩阵
    Gamma = zeros(M * N, M * N);
    for k = 0:M * N - 1
        Gamma(k + 1, k + 1) = exp(-1i * 2 * pi * k / (M * N));
    end

    % 计算 Delta 矩阵
    Delta = zeros(M * N, M * N);
    for k = 0:M * N - 1
        Delta(k + 1, k + 1) = exp(1i * 2 * pi * k / (M * N));
    end

    % 计算 F_MN 和 F_N 矩阵
    F_MN = dftmtx(M * N);
    F_N = dftmtx(N);

    % 构造 Psi
    d_dd = D_dd(:);
    Psi = zeros(M * N, M * N);
    s = kron(F_N' , G_t)*d_dd;
    idx = 1;
    for m = 0:M-1
        for n = 0:N-1
            psi_mn = (F_MN' * (Gamma^m) * F_MN*(Delta^n)) * s;
            Psi(:,idx) = psi_mn;
            idx = idx + 1;
        end
    end
end



function [s] = generate_OTFS_signal(D_dd, M, N, G_t)
    % OTFS 发射机模型
    % D_dd: DD 域数据符号矩阵，大小为 M x N
    % M: 延迟 bins 数量
    % N: 多普勒 bins 数量
    % G_t: 发送脉冲塑形矩阵，大小为 N x N

    % 转换为 TF（时频）域
    F_M = dftmtx(M);
    F_N = dftmtx(N);
    % D_tf = F_M * D_dd * F_N';
    % 
    % % 构造发射信号向量
    % s = reshape(G_t * F_M' * D_tf, [], 1);
    s = kron(F_N', G_t)*D_dd(:);
end

function [H_ni, h_ni] = generate_DD_channel(K, h_k, l_tau, k_nu, M, N)
    % DD 域雷达信道模型
    % K: 目标数量
    % h_k: 目标的增益，大小为 K x 1
    % l_tau: 目标的延迟 bin，大小为 K x 1
    % k_nu: 目标的多普勒 bin，大小为 K x 1
    % M: 延迟 bins 数量
    % N: 多普勒 bins 数量

    % 构造频域对角延迟矩阵
    Gamma = diag(exp(-1j * 2 * pi * (0:(M*N-1)) / (M*N)));

    % 构造频域对角多普勒矩阵
    Delta = diag(exp(1j * 2 * pi * (0:(M*N-1)) / (M*N)));

    % 计算 F_MN 和 F_N 矩阵
    F_MN = dftmtx(M * N);
    
    % 构造雷达信道矩阵
    H_ni = zeros(M*N, M*N);
    h_ni = zeros(M*N, M*N,K );
    for k = 1:K
        % 计算目标的延迟和多普勒
        l_tau_k = l_tau(k); 
        k_nu_k = k_nu(k);   

        % 构造对应目标的矩阵
        H_k = h_k(k) * F_MN' * (Gamma^l_tau_k) * F_MN * (Delta^k_nu_k);
        
        % 更新雷达信道矩阵
        h_ni(:,:,k)= H_k;
        H_ni = H_ni + H_k;
    end
end

function H_n = generate_DD_channel_int(K, h_k, l_tau, k_nu, M, N)
    % DD 域雷达信道模型
    % K: 目标数量
    % h_k: 目标的增益，大小为 K x 1
    % l_tau: 目标的延迟 bin，大小为 K x 1
    % k_nu: 目标的多普勒 bin，大小为 K x 1
    % M: 延迟 bins 数量
    % N: 多普勒 bins 数量

    % 构造频域对角延迟矩阵
    Pi = circshift(eye(M*N),1);

    % 构造频域对角多普勒矩阵
    Delta = diag(exp(1j * 2 * pi * (0:(M*N-1)) / (M*N)));

    % 构造雷达信道矩阵
    H_n = zeros(M*N, M*N);
    for k = 1:K
        % 计算目标的延迟和多普勒
        l_tau_k = floor(l_tau(k)); 
        k_nu_k = floor(k_nu(k));   

        % 构造对应目标的矩阵
        H_k = h_k(k) * (Pi^l_tau_k) * (Delta^k_nu_k);

        % 更新雷达信道矩阵
        H_n = H_n + H_k;
    end
end

function r = receive_signal_non_integer(H_ni, s, v)
    % 考虑实际非整数延迟和多普勒偏移的接收信号模型
    % H_ni: 非整数延迟和多普勒偏移的雷达信道矩阵，大小为 MN x MN
    % s: 发射信号向量，大小为 MN x 1
    % v: 噪声向量，大小为 MN x 1

    r = H_ni * s + v;
end

function r = receive_signal_integer(Psi, h, v)
    % 仅考虑整数延迟和多普勒偏移的接收信号模型
    % Psi: 整数延迟和多普勒偏移的字典矩阵，大小为 MN x MN
    % h: 目标增益向量，大小为 MN x 1
    % v: 噪声向量，大小为 MN x 1

    r = Psi * h + v;
end

function pulse_matrix = gaussian_pulse_matrix(N, M,T)
    % 生成高斯脉冲成型矩阵
    pulse_matrix = zeros(N, M);  % 初始化矩阵

    % 高斯脉冲形状
    t = linspace(-T/2, T/2, M);  % 时间轴
    sigma = T/6;
    gaussian = exp(-t.^2 / (2 * sigma^2));  % 高斯脉冲形状

    pulse_matrix = diag(gaussian);
end

