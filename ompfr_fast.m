function [h_est, support_set] = ompfr_fast(r, Psi, N_iter, epsilon, M, N, G_r, G_t, d_dd)
% OMPFR -  该算法是通过3-step估计，第二次时采用快速网格求解，
% 第三次使用1D-search获得更精确的结果。
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
    Psi_tmp = Psi;
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
        proj = Psi_tmp' * r_n;     
        [~, idx] = max(abs(proj));

        % 更新支撑集
        support_set_int = [support_set_int, idx];

        % 生成分数字典并细化估计
        % 从idx中提取延迟和多普勒的整数部分
        [k_int, l_int] = ind2sub([M, N], idx);
        l_int = l_int-1;
        k_int = k_int-1;
        
        tic;
        % 生成分数字典        
        proj_frac = proj_fractional_dictionary(l_int, k_int, gamma_L, Delta_K,F_MN, F_N,G_r, G_t, d_dd,r_n);
        % 在分数字典上进行投影并找到最大索
        [~, idx_frac] = max(abs(proj_frac));
        [k_frac_ind, l_frac_ind] = ind2sub([M, N], idx_frac);
        l_frac = l_int-1 + (l_frac_ind-1)*2/M;
        k_frac = k_int-1 + (k_frac_ind-1)*2/N;
        [h,phi_refine] = proj_refinement_dictionary(l_frac, k_frac, gamma_L, Delta_K,F_MN, F_N,G_r,G_t,d_dd,r_n);
        disp("ompfr_fast用时：");
        toc;
        % % 细化支撑集和残差的计算
        A = [A,phi_refine];  % 使用分数字典更新近似矩阵
        x = pinv(A)* r;  % 重新计算稀疏向量估计
        % 更新残差
        r_n = r - A * x;
            
        h_est(iter,:) = h;  % 更新稀疏向量估计
        norm(r_n)
        % 检查停止准则
        if norm(r_n) < epsilon
            break;
        end

    end
end

function proj_frac = proj_fractional_dictionary(l_int, k_int, gamma_L, Delta_K,F_MN, F_N,G_r,G_t,d_dd,r_n)
    % 基于整数和分数延迟及多普勒生成分数字典
    % Psi: 整数字典
    % l_int, k_int: 延迟和多普勒的整数部分
    % M, N: 延迟和多普勒的分辨率参数
    % G_t: 发射脉冲形状矩阵
    N = size(F_N,1);
    M = size(F_MN,1)/N;
    proj_frac = zeros(M*N,1);
    index = 1;
    C = (kron(F_N,G_r)*F_MN')'*r_n;
    A = (kron(F_N',G_t)*d_dd)';

    for m = -1:2/M:1-2/M  % 延迟索引范围
        s_k = transpose(A) .* (((gamma_L^(l_int + m)) * F_MN)' * C);
        for n = -1:2/N:1-2/N  % 多普勒索引范围
            nu = (k_int + n);

            % 计算每一列，根据公式(11)
            proj_frac(index) = conj(transpose(diag(Delta_K).^nu))*s_k;
            
            index = index + 1;
        end
    end
end

function [h,phi_refine] = proj_refinement_dictionary(l_frac, k_frac, gamma_L, Delta_K,F_MN, F_N,G_r,G_t,d_dd,r_n)
    % 基于整数和分数延迟及多普勒生成分数字典
    % Psi: 整数字典
    % l_int, k_int: 延迟和多普勒的整数部分
    % M, N: 延迟和多普勒的分辨率参数
    % G_t: 发射脉冲形状矩阵
    N = size(F_N,1);
    M = size(F_MN,1)/N;
    l_tmp = l_frac;
    k_tmp = k_frac;
    % epoint
    C = (kron(F_N,G_r)*F_MN')'*r_n;
    A = (kron(F_N',G_t)*d_dd)';
    % find l_frac & k_frac
    delay_l = exp(-1i * 2 * pi * (0:M * N-1)/(M * N));
    doppler_k = exp(1i * 2 * pi * (0:M * N-1)/(M * N));
    rho = 0.618;    % 黄金分割比例
    N_i = 50;
    for i = 1:N_i
        l_pre = l_tmp;
        k_pre = k_tmp;
        s_l = transpose(A * (F_MN * (Delta_K^k_tmp))').*C;
        l_right = l_tmp+2/(M*N);
        l_left = l_tmp-2/(M*N);
        for l_ind = 1:N_i
            l_d = rho * (l_right - l_left);
            % 初始的两个中间点
            l_x1 = l_right - l_d;
            l_x2 = l_left + l_d;
            % 初始函数值
            f_11 = abs(conj(delay_l.^l_x1)*s_l);
            f_12 = abs(conj(delay_l.^l_x2)*s_l);
            if f_11>f_12
                l_right = l_x2;
            else
                l_left = l_x1;
            end
            if (l_right-l_left)<0.0001
                break;
            end
        end
        l_tmp = (l_right+l_left)/2;
        s_k = transpose(A) .* (((gamma_L^l_tmp) * F_MN)' * C);
        k_right = k_tmp+2/(M*N);
        k_left = k_tmp-2/(M*N);
        for k_ind = 1:N_i
            k_d = rho * (k_right - k_left);
            k_x1 = k_right - k_d;
            k_x2 = k_left + k_d;
            f_21 = abs(conj(doppler_k.^k_x1)*s_k);
            f_22 = abs(conj(doppler_k.^k_x2)*s_k);
            if f_21>f_22
                k_right = k_x2;
            else
                k_left = k_x1;
            end
            if (k_right-k_left)<0.0001
                break;
            end
        end
        k_tmp = (k_right+k_left)/2;
        if abs(l_pre-l_tmp) < 0.001 && abs(k_tmp-k_pre)<0.001
            break;
        end
    end
    % 计算每一列，根据公式(11)
    phi_refine = kron(F_N,G_r)*F_MN' * (gamma_L^l_tmp) * F_MN * (Delta_K^k_tmp) * kron(F_N',G_t)*d_dd;
    h = [l_tmp,k_tmp];
end