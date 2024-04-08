function h = ompfr_g(r_tilde, Phi_tilde, N_ite, epsilon)
    % 初始化
    r = r_tilde;
    h = zeros(size(Phi_tilde, 2), 1);
    I = [];
    Phi = Phi_tilde;
    A = [];
    for n = 1:N_ite
        % 步骤 3: 估计索引
        [~, idx] = max(abs(Phi' * r));
        I(n) = idx;
        % 步骤 4: 更新索引
        l_n = floor((I(n) - 1) / size(Phi, 1)) + 1;
        k_n = mod(I(n) - 1, size(Phi, 1)) + 1;
        % 步骤 5: 计算分数字典
        Phi_f = PhiFractional(Phi, l_n, k_n);
        % 步骤 6: 分数索引估计
        [~, idx_f] = max(abs(Phi_f' * r));
        l_f = floor((idx_f - 1) / size(Phi, 1)) + 1;
        k_f = mod(idx_f - 1, size(Phi, 1)) + 1;
        % 步骤 8: 支持更新
        S_n = [I(1:n), idx_f];
        % 步骤 9: 计算非稀疏元素
        alpha_n = pinv(Phi_f) * r;
        
        % 步骤 10: 更新非稀疏元素向量
        A_n = zeros(size(Phi_tilde, 2), 1);
        A_n(S_n) = alpha_n;
        A = [A, A_n];
        % 步骤 11: 更新残差
        r = r -  A_n* alpha_n;
        % 输出
         h(I) = alpha_n;
        % 步骤 12: 检查终止条件
        if norm(r) <= epsilon
            break;
        end
    end

end

function Psi_frac = generate_fractional_dictionary(Psi, l_int, k_int, M, N)
    % 生成分数字典
    %
    % 参数:
    % Psi : 原始整数字典矩阵
    % l_int : 整数延迟索引
    % k_int : 整数多普勒索引
    % M, N : 延迟和多普勒的分辨率参数
    %
    % 返回:
    % Psi_frac : 分数字典矩阵

    % 初始化分数字典
    Psi_frac = zeros(size(Psi, 1), 2 * M * N);

    % 根据整数索引确定分数延迟和多普勒的范围
    l_range = (l_int - 1):(l_int + 1);
    k_range = (k_int - 1):(k_int + 1);

    % 生成分数字典
    frac_idx = 1;
    for l = l_range
        for k = k_range
            % 对于每个分数延迟和多普勒，生成对应的列
            Psi_frac(:, frac_idx) = generate_fractional_column(Psi, l, k, M, N);
            frac_idx = frac_idx + 1;
        end
    end
end
