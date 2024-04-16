function [h_est, support_set] = omp_fibo(r, Psi, N_iter, epsilon, M, N, G_t, d_dd)
% OMPFR - Orthogonal Matching Pursuit with Fractional Refinement
%
% 参数:
% r : 接收信号向量 (列向量)
% Psi : 整数字典矩阵
% N_iter : 最大迭代次数
% epsilon : 残差阈值
% M, N : 延迟和多普勒的分辨率参数
%
% 返回:
% h_est : 估计的稀疏向量
% support_set : 支撑集，包含每次迭代选中的延迟和多普勒索引

    % 初始化
    r_n = r;                        % 初始化残差
    h_est = zeros(size(Psi, 2), 2); % 估计的稀疏向量
    support_set_int = [];               % 支撑集
    support_set = [];
    A = [];

    gamma_L = diag(exp(-1i * 2 * pi * (0:M*N-1)/(M*N)));  % 延迟相位移动
    Delta_K = diag(exp(1i * 2 * pi * (0:M*N-1)/(M*N)));   % 多普勒相位移动
    F_MN = dftmtx(M*N);
    F_N = dftmtx(N);

    % 迭代过程
    for iter = 1:N_iter
        % 在字典上进行投影并找到最大索引
        proj = Psi' * r_n;     
        [~, idx] = max(abs(proj));

        % 更新支撑集
        support_set_int = [support_set_int, idx];

        % 生成分数字典并细化估计
        % 从idx中提取延迟和多普勒的整数部分
        [k_int, l_int] = ind2sub([M, N], idx);
        l_int = l_int-1;
        k_int = k_int-1;

        indices_part_min = N/2:N:M*N-N/2;

        % 生成最小值字典
        Psi_min_part = l_dictionary(l_int, k_int, gamma_L, Delta_K,F_MN, F_N, G_t, d_dd,indices_part_min);
        proj_min = Psi_min_part(1:size(Psi,1),:)' * r_n; 
        [~, idx_min] = max(abs(proj_min));

        [k_frac_ind, l_frac_ind] = ind2sub([M, N], indices_part_min(idx_min));
        l_frac_m = l_int-1 + (l_frac_ind-1)*2/M;
        k_frac_m = k_int-1 + (k_frac_ind-1)*2/N;
        % 黄金分割比一维搜索
        l_left = l_frac_m - 0.25;
        l_right = l_frac_m + 0.25;
        k_left = k_frac_m - 0.25;
        k_right = k_frac_m + 0.25;
        N_i = 50;
        fi = fib(N_i+3);
        for i = 1:N_i
            rho = 1;    % 黄金分割比例
           
            l_d = rho * (l_right - l_left);
            k_d = rho * (k_right - k_left);
            
            % 初始的两个中间点
            l_x1 = l_left + fi(i+1)/fi(i+3) * l_d;
            l_x2 = l_left + fi(i+2)/fi(i+3) * l_d;
            k_x1 = k_left + fi(i+1)/fi(i+3) * k_d;
            k_x2 = k_left + fi(i+2)/fi(i+3) * k_d;

            
            % 初始函数值
            phi_11 = F_MN' * (gamma_L^l_x1) * F_MN * (Delta_K^k_x1) * kron(F_N',G_t)*d_dd;
            phi_12 = F_MN' * (gamma_L^l_x1) * F_MN * (Delta_K^k_x2) * kron(F_N',G_t)*d_dd;
            phi_21 = F_MN' * (gamma_L^l_x2) * F_MN * (Delta_K^k_x1) * kron(F_N',G_t)*d_dd;
            phi_22 = F_MN' * (gamma_L^l_x2) * F_MN * (Delta_K^k_x2) * kron(F_N',G_t)*d_dd;

            f11 = phi_11'*r_n;
            f12 = phi_12'*r_n;
            f21 = phi_21'*r_n;
            f22 = phi_22'*r_n;
            
            [~,ind] = max([f11,f12,f21,f22]);
            if ind == 1
                l_right = l_x2;
                k_right = k_x2;
                phi = phi_11;
            elseif ind == 2
                l_right = l_x2;
                k_left = k_x1;
                phi = phi_12;
            elseif ind == 3
                l_left = l_x1;
                k_right = k_x2;
                phi = phi_21;
            elseif ind == 4
                l_left = l_x1;
                k_left = k_x1;
                phi = phi_22;
            end
            if (l_right-l_left)<0.0001 && (k_right-k_left)<0.0001
                break;
            end
        end
        l_frac = (l_left+l_right)/2;
        k_frac = (k_left+k_right)/2;
        A = [A,phi];  % 使用分数字典更新近似矩阵
        x = pinv(A)* r;  % 重新计算稀疏向量估计
        % 更新残差
        r_n = r - A * x;
        
        % h_est_rel(iter,:) = [l_frac_rel, k_frac_rel];
        h_est(iter,:) = [l_frac, k_frac];  % 更新稀疏向量估计
        norm(r_n)
        % 检查停止准则
        if norm(r_n) < epsilon
            break;
        end
    end

end

function b = fib(k)
    b(1) = 1;
    b(2) = 1;
    for i = 3:k
        b(i) = b(i-1)+b(i-2);
    end
end
function Psi_min = l_dictionary(l_int, k_int, gamma_L, Delta_K,F_MN, F_N,G_t,d_dd,indices)
    N = size(F_N,1);
    M = size(F_MN,1)/N;
    
    Psi_min = zeros(M*N, M);
    index = 1;

    for index = 1:length(indices)  % 多普勒索引范围

        % 延迟和多普勒的总值
        tau = l_int -1 + (index-1)*2/M ;
        nu = (k_int -1 + (indices(1)-1)*2/N);

        % 计算每一列，根据公式(11)
        Psi_min(:, index) = F_MN' * (gamma_L^tau) * F_MN * (Delta_K^nu) * kron(F_N',G_t)*d_dd;

    end
end
function Psi_min = k_dictionary(l_int, k_int, gamma_L, Delta_K,F_MN, F_N,G_t,d_dd,indices)
    N = size(F_N,1);
    M = size(F_MN,1)/N;
    
    Psi_min = zeros(M*N, M);
    index = 1;

    for index = 1:length(indices)  % 多普勒索引范围

        % 延迟和多普勒的总值
        tau = l_int -1 + (indices(1)-1)*2/M ;
        nu = (k_int -1 + (index-1)*2/N);

        % 计算每一列，根据公式(11)
        Psi_min(:, index) = F_MN' * (gamma_L^tau) * F_MN * (Delta_K^nu) * kron(F_N',G_t)*d_dd;

    end
end