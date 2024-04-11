function [h_est, support_set] = omp_fast(r, Psi, N_iter, epsilon, M, N, G_t, d_dd)
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
        
        tic;
        % 生成小字典        
        Psi_part = partial_dictionary(l_int, k_int, gamma_L, Delta_K,F_MN, F_N, G_t, d_dd);
        % 
        proj_part = Psi_part(1:size(Psi,1),:)' * r_n;
        [~, idx_part] = max(abs(proj_part));
        indices_part_min = mod(idx_part, N):N:(M*N-N+mod(idx_part,N));

        % 生成最小值字典
        Psi_min_part = min_dictionary(l_int, k_int, gamma_L, Delta_K,F_MN, F_N, G_t, d_dd,indices_part_min);


        % % 相关
        % for i = 1:size(Psi_part,2)
        %     A_corr = [A,Psi_part(1:size(Psi,1), i)];
        %     x_corr = pinv(A_corr)*r;
        %     r_corr = r-A_corr*x_corr;
        %     r_val(i) = norm(r_corr);
        % end
        % [~,idx_part_res] = min(r_val);
        % indices_res_min = mod(idx_part_res, N):N:(M*N-N+mod(idx_part_res,N));



        % 在最小字典上进行投影并找到最大索引
        proj_min = Psi_min_part(1:size(Psi,1),:)' * r_n; 
        [~, idx_min] = max(abs(proj_min));

        [k_frac_ind, l_frac_ind] = ind2sub([M, N], indices_part_min(idx_min));
        l_frac = l_int-1 + (l_frac_ind-1)*2/M;
        k_frac = k_int-1 + (k_frac_ind-1)*2/N;
        % find the neighbour to get rid of the effect of other objects.
        for l = 1:3
            for k = 1:2:3
                Psi_neigh(:,(l-1)*2+(k+1)/2) = F_MN' * (gamma_L^(l_frac+2/M*(l-2))) * F_MN * (Delta_K^(k_frac+2/N*(2*k-3))) * kron(F_N',G_t)*d_dd;
            end
        end
        Psi_min_neigh = [Psi_neigh,Psi_min_part(:,idx_min)];
        proj_neigh = [Psi_min_neigh(1:size(Psi,1),:)]' * r_n;
        [~,idx_neigh] = max(abs(proj_neigh));
        if idx_neigh~=7
            [k_neigh, l_neigh] = ind2sub([2, 3], idx_neigh);
            l_frac = l_frac+2/N*(l_neigh-2);
            k_frac = k_frac+2/N*(2*k_neigh-3);
        end
        
        disp('omp_fast用时:');
        toc;

        % iter
        indices_min_iter(iter,:) = indices_part_min; 
        Psi_iter(:,:,iter) = Psi_min_part(1:size(Psi,1), :);
        if iter~= 1
            for i = 1:M
                for j =1:M
                    A_iter = [Psi_iter(:,i,1),Psi_iter(:,j,2)];
                    x_iter = pinv(A_iter)*r;
                    r_iter = r-A_iter*x_iter;
                    val_iter(i,j) = norm(r_iter); 
                end
            end
        end

        A = [A,Psi_min_neigh(1:size(Psi,1), idx_neigh)];  % 使用分数字典更新近似矩阵
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

function Psi_part = partial_dictionary(l_int, k_int, gamma_L, Delta_K,F_MN, F_N,G_t,d_dd)
    % 基于整数和分数延迟及多普勒生成分数字典
    % Psi: 整数字典
    % l_int, k_int: 延迟和多普勒的整数部分
    % M, N: 延迟和多普勒的分辨率参数
    % G_t: 发射脉冲形状矩阵
    N = size(F_N,1);
    M = size(F_MN,1)/N;
    
    Psi_part = zeros(M*N, N);
    index = 1;
    for n = -1:2/N:1-2/N  % 多普勒索引范围

        % 延迟和多普勒的总值
        tau = l_int;
        nu = (k_int + n);

        % 计算每一列，根据公式(11)
        Psi_part(:, index) = F_MN' * (gamma_L^tau) * F_MN * (Delta_K^nu) * kron(F_N',G_t)*d_dd;
        index =index+1;
    end
end

function Psi_min = min_dictionary(l_int, k_int, gamma_L, Delta_K,F_MN, F_N,G_t,d_dd,indices)
    % 基于整数和分数延迟及多普勒生成分数字典
    % Psi: 整数字典
    % l_int, k_int: 延迟和多普勒的整数部分
    % M, N: 延迟和多普勒的分辨率参数
    % G_t: 发射脉冲形状矩阵
    N = size(F_N,1);
    M = size(F_MN,1)/N;
    
    Psi_min = zeros(M*N, N);
    index = 1;

    for index = 1:length(indices)  % 多普勒索引范围

        % 延迟和多普勒的总值
        tau = l_int -1 + (index-1)*2/M ;
        nu = (k_int -1 + (indices(1)-1)*2/N);

        % 计算每一列，根据公式(11)
        Psi_min(:, index) = F_MN' * (gamma_L^tau) * F_MN * (Delta_K^nu) * kron(F_N',G_t)*d_dd;

    end
end

function relevant_indices = get_relevant_indices(idx, N, matrix_size)
    div = 4;
    % 获取包括该列周围共 N/4 列的列序号
    relevant_indices = [];

    % 获取每隔 N 列共 N/4 个列序号
    gap_indices = max(mod(idx-N/div*N,N), idx - N/div*N):N:min(idx+(N/div-1)*N, matrix_size-N+mod(idx,N));
    for i = 1:length(gap_indices)
        % 计算边界值
        left_boundary = max(1, gap_indices(i) - N/div);
        right_boundary = min(matrix_size, gap_indices(i) + N/div-1);
        % 添加左侧列序号到结果集合
        relevant_indices = [relevant_indices, left_boundary:right_boundary];
    end

end

