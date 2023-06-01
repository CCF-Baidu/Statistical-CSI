clc;clear all;close all;

M = 5;
K = 15;
M_h = 8;
M_v = 8;
L = M_h * M_v;


N0 = 1e-10;

ref = 1/(1e-12);
ref1 = 1/(1e-10);

N0 = N0 * ref1; %;

W = ones(K, 1)/10; %0.1W

gamma_h = 1 * ref1; %;
gamma_v = 1 * ref1^(0.5); %;

opt_beta = ones(K, 1)/K; %0.1W

alpha_UI = 2.8;
alpha_IB = 2.2;
alpha_UB = 3.5;
PL_0 = 10^(-30/10); %dB the channel gain at the reference distance


Rican_UI   = 3 + sqrt(12); % device-RIS channel Rician factor
Rican_UB   = 3 + sqrt(12); % direct channel Rician factor

MENT = 10;

BS_loc = [0,0,0]; %位于XoZ面上
RIS_loc = [0,40,3];%位于YoZ面上

load User_loc.mat

d_IB = norm(BS_loc -  RIS_loc);
pathloss_IB=sqrt(PL_0*(d_IB)^(-alpha_IB));    % Large-scale pass loss from the BS to the IRS

for i = 1:K
    d_UI(i) = norm(RIS_loc - User_loc(i,:)); % distance from the user to the IRS
    d_UB(i) = norm(BS_loc - User_loc(i,:)); % distance from the user to the BS
    pathloss_UB(i)=sqrt(PL_0*(d_UB(i))^(-alpha_UB));    % Large-scale pass loss from the user to the BS
    pathloss_UI(i)=sqrt(PL_0*(d_UI(i))^(-alpha_UI));    % Large-scale pass loss from the user to the IRS
end

% A = [BS_loc;
%  RIS_loc;
% Location1;
% Location2
% ];
% [mm,nn] = size(A);
% figure(1);
% hold on;
% grid on;
%     plot3(A(:,1),A(:,2),A(:,3),'o')   ;
% for i = 1:mm
%     X1 = [A(i,1);0];
%     Y1 = [A(i,2);0];
%     Z1 = [A(i,3);0];
%     plot3(X1,Y1,Z1)
%     %text(A(i,1),A(i,2),A(i,3),['(' num2str(A(i,1)) ',' num2str(A(i,2)) ',' num2str(A(i,3)) ')'])
% end
% view(-20,40)
% xlabel( 'x');
% ylabel( 'y');
% zlabel( 'z');

%% generate channel IRS-BS
kd = pi;


A_ula_BS = @(x) exp( (0:M-1)'*1j*kd*sin(x(1))*sin(x(2)));

aoa =zeros(2,1);  % x(1)---azimuth, x(2)---elevation
aoa(1) = pi/2; % x(1)---azimuth
aoa(2) = atan((BS_loc(2)-RIS_loc(2))/(BS_loc(3)-RIS_loc(3))); % x(2)---elevation

aod =zeros(2,1);  % x(1)---azimuth
aod(1) = pi/2; % x(1)---azimuth
aod(2) = pi/2 - aoa(2);

AT_G_bs = A_ula_BS(aoa);
for i=1:M_h
    for j=1:M_v
        AR_G_irs(i + (j - 1) * M_h, 1) = exp(1j*kd*((i - 1)*sin(aod(1))*sin(aod(2)) + (j - 1) * cos(aod(2))));
    end
end

G_los = AT_G_bs*AR_G_irs' * ref1^(0.25); %; % Tx-antenn gain BS

G = pathloss_IB * G_los;

g1 = pathloss_IB;

%% generate channel User-BS
kd = pi;

A_ula_BS = @(x) exp( (0:M-1)'*1j*kd*sin(x(1))*sin(x(2)));

aoa =zeros(2, 1);  % x(1)---azimuth, x(2)---elevation
for i = 1:K
    U_loc = User_loc(i, :);
    aoa(1) = atan((BS_loc(2)-U_loc(2))/(BS_loc(1)-U_loc(1))); % x(1)---azimuth
    aoa(2) = pi/2; % x(2)---elevation
    AT_hd_bs = A_ula_BS(aoa);
    h_d_los(:, i) = AT_hd_bs * ref1^(0.5); %; % Tx-antenn gain BS
    h_d_nlos(:, i)=sqrt(1/2).*(randn(M,1)+1j.*randn(M,1)) * ref1^(0.5); %;
    h_d(:, i) = pathloss_UB(i) * (sqrt(Rican_UB/(1+Rican_UB))*h_d_los(:, i)+ sqrt(1/(1+Rican_UB))*h_d_nlos(:, i));
    h1(i) = pathloss_UB(i) * sqrt(Rican_UB/(1+Rican_UB));
    h2(i) = pathloss_UB(i) * sqrt(1/(1+Rican_UB));
end



%% generate channel User-IRS
kd = pi;

aoa =zeros(2,1);  % x(1)---azimuth, x(2)---elevation

for ii = 1:K
    U_loc = User_loc(ii, :);
    aoa(1) = atan((RIS_loc(2) - U_loc(2))/(RIS_loc(1) - U_loc(1))); % x(1)---azimuth
    aoa(2) = pi - acos((RIS_loc(3) - U_loc(3))/d_UI(ii)); % x(1)---azimuth; % x(2)---elevation
    for i=1:M_h
        for j=1:M_v
            AR_G_irs(i + (j - 1) * M_h, 1) = exp(1j*kd*((i - 1)*sin(aod(1))*sin(aod(2)) + (j - 1) * cos(aod(2))));
        end
    end
    h_r_los(:,ii) = AR_G_irs * ref1^(0.25); %; 
    h_r_nlos(:, ii)=sqrt(1/2).*(randn(L,1)+1j.*randn(L,1)) * ref1^(0.25); %;
    h_r(:, ii) = pathloss_UI(ii) * (sqrt(Rican_UI/(1+Rican_UI))*h_r_los(:, ii)+ sqrt(1/(1+Rican_UI))*h_r_nlos(:, ii));
    v1(ii) = pathloss_UI(ii) * sqrt(Rican_UI/(1+Rican_UI));
    v2(ii) = pathloss_UI(ii) * sqrt(1/(1+Rican_UI));
end



n_b = 600; %(sample),此种设定下，n_b = 500 Ek约等于0.003（最大的f_k,最小的f_k为1e-5次方级别），T_k约等于0.006最大的f_k,最小的f_k为0.0377）
B_t = 25000000;  % 20mhz,总带宽
f_min = 200000000; %0.2ghz  
f_max = 2000000000; %2ghz 
c_k = 10000 + 20000 * rand(1, K); % the number of CPU cycles required for computing one sample data at user k (cycles/sample)
eta_k = 2 * 1e-28; %The effective switched capacitance in local computation
E_max = 0.1; %energy requirement (J) 
t_max = 0.3; %delay requirement (s)
net_total_params = 21880 * 16;%;/1024; %模型大小(kbit)
opt_pou = ones(K, 1);

f = (f_min + (f_max-f_min) *  rand(K, 1)) .* ones(K, 1);


%%%%%%%%%%%%%%%%%%%%%B1 and B2%%%%%%%%%%%%%%%%%%%%%%
beta_ur = ones(K, 1)/K; %0.1W
d_total = sum(d_UB);
s_total = 1 / d_total;

for i=1:K
    beta_pr(i) = s_total * d_UB(i);
end

for i = 1:K
    G_c_nr(:, i) = h_d(:, i);
    snr_nr(i) = (W(i) * (norm(G_c_nr(:, i)))^2)/N0;
    snr_DB_nr(i) = 10 * log10(snr_nr(i));
    cap_nr(i) = beta_pr(i) * B_t * log2(1 + snr_nr(i));
    r_rand(i) = max(net_total_params/(t_max - c_k(i) * (n_b/f(i))), net_total_params * W(i)/(E_max - (1 / 2) * eta_k * c_k(i) * n_b * f(i)^2));
end

out_flag_nr = cap_nr < r_rand

theta=exp(1j.*rand(L,1).*2.*pi);

for i = 1:K
    G_c_H(:, :, i) = G_los * diag(h_r_los(:, i));
    G_c(:, i) = G * diag(h_r(:, i)) * theta + h_d(:, i);
    snr(i) = (W(i) * (norm(G_c(:, i)))^2)/N0;
    snr_DB(i) = 10 * log10(snr(i));
    cap_rand_ur(i) = beta_ur(i) * B_t * log2(1 + snr(i));
    cap_rand_pr(i) = beta_pr(i) * B_t * log2(1 + snr(i));
    r_rand(i) = max(net_total_params/(t_max - c_k(i) * (n_b/f(i))), net_total_params * W(i)/(E_max - (1 / 2) * eta_k * c_k(i) * n_b * f(i)^2));
end

out_flag_ur = cap_rand_ur < r_rand
out_flag_pr = cap_rand_pr < r_rand

for k=1:K
    G_cs(:, :, k) = G_los * diag(h_r_los(:, k));
    E_c(:, :, k)=[((pathloss_IB)^2) * ((pathloss_UI(k))^2) * (Rican_UI/(1+Rican_UI)) * (G_cs(:, :, k))' * G_cs(:, :, k),sqrt(((pathloss_IB)) * ((pathloss_UI(k))) * ((pathloss_UB(k))))* (Rican_UI/(1+Rican_UI)) * (G_cs(:, :, k))' * h_d_los(:, k); sqrt(((pathloss_IB)) * ((pathloss_UI(k))) * ((pathloss_UB(k)))) * (h_d_los(:, k))' * G_cs(:, :, k), 0 ]; 
end 

Es = randn(L+1,L+1)+1i*randn(L+1,L+1); Es=Es*Es';
[us,~,~] = svd(Es);
Es_partial = us(:,1)*us(:,1)';

obj0 = 0;
maxiters = 30;
for iter1 = 1:maxiters
    cvx_solver sdpt3
    cvx_precision best
    cvx_save_prefs
    cvx_begin 
        variable Es(L+1,L+1) hermitian semidefinite

        rates = 0;
        for k=1:K
            rates = rates + log(1 + (W(k)/N0) * real(trace(E_c(:, :, k) * Es) + ((pathloss_IB)^2) * ((pathloss_UI(k))^2) * (1/(1+Rican_UI)) * trace(G_los' * G_los) + (((pathloss_UB(k)))^2) * (1/(1+Rican_UI)) * L +  (((pathloss_UB(k)))^2) * (Rican_UI/(1+Rican_UI)) * (h_d_los(:, k))' * h_d_los(:, k)))/log(2);
        end 

        maximize rates - 10 * (real(trace((eye(L+1)-Es_partial')* Es)))

        subject to

            diag(Es) ==1;  

    cvx_end
    


    errs(iter1) = abs(cvx_optval-obj0);

    [us,~,~] = svd(Es);
    Es_partial = us(:,1)*us(:,1)';
    obj0 = cvx_optval;
    objs(iter1) = cvx_optval;

end

[u,~,~] = svd(Es);
v_tilde = u(:,1);
vv=v_tilde(1:L)/v_tilde(L+1);
thetas = vv./abs(vv);

for i = 1:K
    G_cs = G * diag(h_r(:, i)) * thetas + h_d(:, i);
    snrs(i) = (W(i) * (norm(G_cs))^2)/N0;
    snr_DBs(i) = 10 * log10(snrs(i));
    cap_rand_rate(i) = beta_ur(i) * B_t * log2(1 + snrs(i));
end

out_flag_sumrate = cap_rand_rate < r_rand

for i = 1:K
    G_cspr = G * diag(h_r(:, i)) * thetas + h_d(:, i);
    snrspr(i) = (W(i) * (norm(G_cspr))^2)/N0;
    snr_DBspr(i) = 10 * log10(snrspr(i));
    cap_rand_ratepr(i) = beta_pr(i) * B_t * log2(1 + snrs(i));
end

out_flag_sumratepr = cap_rand_ratepr < r_rand




G_LOS = G_c_H;
h_UR_LOS = h_r_los;
h_LOS = h_d_los;
H_RB_LOS = G_los;

iter_max = 15;
iter_max1 = 1;
%%%Step I: given pou, theta, update x, y, residual%%%

Phi = diag(theta);
for k=1:K
    E_a(:, :, k)=[(v1(k))^2 *(g1)^2 * (G_LOS(:, :, k))' * G_LOS(:, :, k), v1(k) * g1 * h1(k) * (G_LOS(:, :, k))' * h_LOS(:, k); v1(k) * g1 * h1(k) * (h_LOS(:, k))' * G_LOS(:, :, k), 0 ]; 
    gamma = (v2(k))^2 *(g1)^2 * gamma_v * H_RB_LOS * (H_RB_LOS)' + (h2(k))^2 * gamma_h * eye(M,M);
    Gamma(:,:, k) = gamma;
    E_b(:, :, k)=[2 * (v1(k))^2 *(g1)^2 * (G_LOS(:, :, k))' * Gamma(:, :, k) * G_LOS(:, :, k), 2 * v1(k) * g1 * h1(k) * (G_LOS(:, :, k))' * Gamma(:, :, k) * h_LOS(:, k); 2 * v1(k) * g1 * h1(k) * (h_LOS(:, k))' * Gamma(:, :, k) * G_LOS(:, :, k), 0 ];
    mu(:, k) = v1(k) * g1 * H_RB_LOS * Phi * h_UR_LOS(:, k) + h1(k) * h_LOS(:, k);
end 

for k=1:K
      r_low(k) = max(net_total_params/(t_max - c_k(k) * (n_b/f(k))), net_total_params * W(k)/(E_max - (1 / 2) * eta_k * c_k(k) * n_b * f(k)^2));
      cmin(k) = (N0/W(k)) * (2^(r_low(k)/(opt_beta(k) * B_t)) - 1);
      c(k) = cmin(k);
      [T1,T2]=eig(-Gamma(:, :, k));

      if max(real(diag(T2)))>=0
          y(k)=max(real(diag(T2)));
      else
          y(k)=0;
      end
end

for k=1:K
     element_1=vec(Gamma(:, :, k));
     element_2=sqrt(2)*((Gamma(:, :, k))^0.5)' * mu(:, k);
     x_ini(k) =  norm([element_1;element_2],2);
end

for iter=1:iter_max

    E = Es; E=E*E';
    [u,~,~] = svd(E);
    E_partial = u(:,1)*u(:,1)';

    obj0 = 0;
    maxiter = 20;
    for iter1 = 1:maxiter
        cvx_solver sdpt3
        cvx_precision best
        cvx_save_prefs
        cvx_begin 
            variable E(L+1,L+1) hermitian semidefinite
            variable opt_beta(K)
            variable x(K) 
            variable residual(K) 
            expressions   constant(K) constraint_a1(K) constraint_a2(K) constraint_a3(K) constraint_a4(k);
            beta_sum = 0;
            for k=1:K
                constraint_a1(k) = real(trace(Gamma(:, :, k))-(sqrt(2*log(1/opt_pou(k))))*x(k) + (log(opt_pou(k)))*y(k) + trace(E_a(:, :, k)*E)) + (h1(k))^2 * (norm(h_LOS(:, k)))^2- (N0/W(k)) * (exp((log(2)*r_low(k)/B_t) * inv_pos(opt_beta(k))) - 1) -residual(k);%2^x can be re-written as exp(x*log(2))
                constraint_a2(k) = norm(Gamma(:, :, k),'fro')^2 + trace(E_b(:, :, k)*E) + 2 * (h1(k))^2 * (norm(((Gamma(:, :, k))^0.5)' * h_LOS(:, k)))^2 -(2*real(x_ini(k)*x(k))-x_ini(k)^2);  
                beta_sum = beta_sum + opt_beta(k);
            end 
    
            maximize sum(residual) - 5 * (real(trace((eye(L+1)-E_partial')* E)))
    
            subject to

                residual>=0;

                real(constraint_a1)>=0;
                real(constraint_a2)<=0;
                diag(E) ==1;  
                x>=0;
                beta_sum<=1;
        cvx_end
      
        err(iter1) = abs(cvx_optval-obj0);

        [u,~,~] = svd(E);
        E_partial = u(:,1)*u(:,1)';

        obj0 = cvx_optval;

        obj2(iter1) = cvx_optval;
    end

    [u,~,~] = svd(E);
    v_tilde = u(:,1);
    vv=v_tilde(1:L)/v_tilde(L+1);
    theta = vv./abs(vv);
    % theta = E(1:L,L+1);
    


    for i = 1:K
        G_c_opt_iter = G * diag(h_r(:, i)) * theta + h_d(:, i);
        snr_opt_iter = (W(i) * (norm(G_c_opt_iter))^2)/N0;
        snr_DB_iter(i) = 10 * log10(snr_opt_iter);
    end
    
    
    mean(snr_DB)
    mean(snr_DB_iter)
    mean_snr_DB_iter(iter) = mean(snr_DB_iter);

    for k=1:K
        r_low(k) = max(net_total_params/(t_max - c_k(k) * (n_b/f(k))), net_total_params * W(k)/(E_max - (1 / 2) * eta_k * c_k(k) * n_b * f(k)^2));
    end 
    Phi = diag(theta);
    for k=1:K
        mu(:, k) = v1(k) * g1 * H_RB_LOS * Phi * h_UR_LOS(:, k) + h1(k) * h_LOS(:, k);

        c(k) = (N0/W(k)) * (exp((log(2)*r_low(k)/B_t)/opt_beta(k)) - 1);
    end 



    %%Step III: given  x, y, theta, update pou%%%
    
    cvx_solver sdpt3
    cvx_precision best
    cvx_save_prefs
    
    cvx_begin 
        variable t(K) 
        % variable c(K)

        t_sum = 0;
        for k=1:K
            t_sum = t_sum + exp(-1 * t(k)); % t(k)-5次方级
        end  

        
        minimize t_sum
      
        subject to
             for k=1:K
                  2 * ((x(k))^2) * t(k) + 2 * sqrt(2) * ((t(k))^1.5) * x(k) * y(k) + ((t(k))^2) <= (real(trace(Gamma(:, :, k)) +  (mu(:, k))' * (mu(:, k))) -  c(k))^2;
             end 
             t>=0;
    cvx_end

    for k=1:K
        pou(k) = exp(-1 * t(k));
    end  

    opt_pou = pou;
    
    obj(iter) = sum(pou);

end

for i = 1:K
    G_c_H_opt(:, :, i) = G_los * diag(h_r_los(:, i));
    G_c_opt(:, i) = G * diag(h_r(:, i)) * theta + h_d(:, i);
    snr_opt(i) = (W(i) * (norm(G_c_opt(:, i)))^2)/N0;
    cap_opt(i) = opt_beta(i) * B_t * log2(1 + snr_opt(i));
    snr_DB_opt(i) = 10 * log10(snr_opt(i));
end

out_flag_opt = cap_opt < r_low

