function h_est = msp(r, Phi, epsilon)
    k = 0;
    k_max = 8;
    proj = Phi'*r;
    [val, idx] = max(abs(proj));
    r_norm = 0;
    for i = 1:length(r)
        S_k = idx(1:i);
        ai = pinv(Phi(:,Si))*r;
        r_k = r - Phi(:, Si)*ai;
        k = 0;
        % sp
        while k<k_max
            k = k+1;
            proj_k = Phi'*r_k;      % 残差投影
            [val_k, idx_k] = max(abs(proj_k));
            S_hat = unique([S_k, idx_k(1:i)]);  
            z = Phi(:,S_hat)*r;
            [val_z, idx_z] = max(abs(z));
            S_k = idx_z(1:i);
            a_k = pinv(Phi(:,S_k))*r
            r_k = y - Phi(:,S_k)*a_k;
        end
        ri = r_k;
        if norm(ri)-r_norm <epsilon
            break;
        end
        r_norm = norm(ri);
        
        Si = S_k;
    end
    h_est = a_k;
end