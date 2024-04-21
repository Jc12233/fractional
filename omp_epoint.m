function [h_est, support_set] = omp_epoint(r, Psi, N_iter, epsilon, M, N, G_t, d_dd)
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
    Phi_sel = [];
    G_r = G_t;

    gamma_L = diag(exp(-1i * 2 * pi * (0:M*N-1)/(M*N)));  % 延迟相位移动
    Delta_K = diag(exp(1i * 2 * pi * (0:M*N-1)/(M*N)));   % 多普勒相位移动
    log_ga = log(diag(gamma_L));
    log_de = log(diag(Delta_K));
    F_MN = dftmtx(M*N);
    F_N = dftmtx(N);
    N_i = 50;
    fi = fib(N_i+3);

    % 迭代过程
    for iter = 1:N_iter
        % 在字典上进行投影并找到最大索引
        proj = Psi' * r_n;     
        [~, idx] = max(abs(proj));

        % 更新支撑集
        % support_set_int = [support_set_int, idx];

        % 生成分数字典并细化估计
        % 从idx中提取延迟和多普勒的整数部分
        [k_int, l_int] = ind2sub([M, N], idx);
        l_int = l_int-1;
        k_int = k_int-1;
        
        l_tmp = l_int;
        k_tmp = k_int;
        tic;
        for ind = 1:5
            % epoint
            C = (kron(F_N,G_r)*F_MN')'*r_n;
            A = (kron(F_N',G_t)*d_dd)';
            % find l_frac & k_frac
            s_l = transpose(A * (F_MN * (Delta_K^k_tmp))').*C;
            s_k = transpose(A) .* (((gamma_L^l_tmp) * F_MN)' * C);
            delay_l = exp(-1i * 2 * pi * (0:M * N-1)/(M * N));
            doppler_k = exp(1i * 2 * pi * (0:M * N-1)/(M * N));
            l_right = l_tmp+1/ind;
            l_left = l_tmp-1/ind;
            k_right = k_tmp+1/ind;
            k_left = k_tmp-1/ind;
    

    
            for i = 1:N_i
                rho = 0.618;    % 黄金分割比例
               
                l_d = rho * (l_right - l_left);
                k_d = rho * (k_right - k_left);
                
                % 初始的两个中间点
                l_x1 = l_right - l_d;
                l_x2 = l_left + l_d;
                k_x1 = k_right - k_d;
                k_x2 = k_left + k_d;
    
                
                % 初始函数值
                f_11 = abs(conj(delay_l.^l_x1)*s_l);
                f_12 = abs(conj(delay_l.^l_x2)*s_l);
                f_21 = abs(conj(doppler_k.^k_x1)*s_k);
                f_22 = abs(conj(doppler_k.^k_x2)*s_k);
    
                
                if f_11>f_12
                    l_right = l_x2;
                else
                    l_left = l_x1;
                end
                if f_21>f_22
                    k_right = k_x2;
                else
                    k_left = k_x1;
                end
                if (l_right-l_left)<0.0001 && (k_right-k_left)<0.0001
                    break;
                end
            end

            l_frac = (l_left+l_right)/2;
            k_frac = (k_left+k_right)/2;
            l_tmp = l_frac;
            k_tmp = k_frac;
        end
        toc;
        phi = kron(F_N,G_r)*F_MN' * (gamma_L^l_frac) * F_MN * (Delta_K^k_frac) * kron(F_N',G_t)*d_dd;
        Phi_sel = [Phi_sel,phi];  % 使用分数字典更新近似矩阵
        x = pinv(Phi_sel)* r;  % 重新计算稀疏向量估计
        % 更新残差
        r_n = r - Phi_sel * x;
        
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
