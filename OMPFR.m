function h = OMPFR(r_n, r, phi, N_ite, tol, M, N, d_dd, G)
    % 初始化
    F_MN = dftmtx(M*N);
    F_MN_conj_trans = F_MN';
    F_N = dftmtx(N);
    F_N_conj_trans = F_N';
    z = exp(1i*2*pi/M/N);
    doppler_vector = z.^(0:M*N-1);
    doppler_matrix = diag(doppler_vector);
    TAO = conj(doppler_matrix);
    m = length(r);
    r_temp = zeros(N_ite, m);
    r_temp(1,:) = r;
    h = zeros(m, 1);
    I = zeros(N_ite, 1);
    phi0 = phi;
    gama = [];
    A = [];
    for n = 1:N_ite
        I(n) = argmax(abs(phi0'*r_temp(n,:)));
        l = floor(I(n)/N);
        k = mod(I(n), N);
        phi_f = F_MN_conj_trans * (TAO^(l-1+2*(m-1)/M)) * F_MN * ...
                (doppler_matrix^(k-1+2*(n-1)/N)) * kron(F_N_conj_trans,G) * d_dd;
        I_f = argmax(abs(phi_f'*r_temp(n,:)));
        lf = floor(I_f/N);
        kf = mod(I_f, N);
        gama = [gama, I_f];
        alpha = pinv(phi_f) * r_n;
        A = [A,alpha];
        r_temp(n+1,:) = r_temp(n,:) - phi_f * alpha;
        if norm(r_temp(n+1,:)) < tol
            break;
        end
    end
    h(gama) = A(gama);
end
