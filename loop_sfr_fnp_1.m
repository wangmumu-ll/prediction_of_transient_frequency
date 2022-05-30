clc
clear all

% column_name = ['SFR最大频差' 'SFR频率最低点时间' 'FNP最大频差' 'FNP频率最低点时间'];
column_name = ["Pd" "H" "R" "D" "TR" "FH" "dfmax_SFR" 'tnadir_SFR' 'dfmax_FNP' 'tnadir_FNP';]
writematrix(column_name, 'sfr_fnp_data_1.csv'); %写入数据 第一版数据两个输入两个输出

%% 设置固定值的参数
Sb = 100;           %定义容量基准值-固定值
SN = 567.5;         %系统总容量-固定值
Km = 1;             %机械功率增益系数-固定值
t=0:0.01:5;         %仿真时间_固定值

%% 设置具有区间范围的参数
%产生一个a至b之间的随机矩阵，大小为1x5；
%a + (b-a) * rand(1,5); 如：a,b = 2,5
%s5 = 2 + (5-2) * rand(1,5);
Pd = -0.5 + (-1.5+0.5) * rand(1,100);          %扰动功率50MW/0.5-1.5
H  = (3 + (9-3) * rand(1,100)) *SN/Sb;   %发电机的总惯性时间常数33.0512 /3-9
R  = (0.04 + (0.1-0.04) * rand(1,100)) *Sb/SN;    %调速器的频率调差系数0.008810 /0.04-0.1
D  = (0 + (2-0) * rand(1,100));             %发电机的等效阻尼系数/0-2
%D = 0;
TR = (6 + (14-6) * rand(1,100));             %原动机的再热时间常数/6-14
FH = (0.15 + (0.4-0.15) * rand(1,100));           %汽轮机的高压杠功率系数/0.15-0.4

disp('loop start');

%% 循环生成样本数据
for i = 1:10000
     Pd_one = Pd(1, randi([1,100]));          %扰动功率
     H_one  = H(1, randi([1,100]));    %发电机的总惯性时间常数
     R_one  = R(1, randi([1,100]));    %调速器的频率调差系数
     D_one  = D(1, randi([1,100]));    %发电机的等效阻尼系数
     TR_one = TR(1, randi([1,100]));   %原动机的再热时间常数
     FH_one = FH(1, randi([1,100]));   %汽轮机的高压杠功率系数
     new_SFR_FNP(Sb, SN, Pd_one, H_one, R_one, D_one, TR_one, FH_one, Km, t);
     %new_SFR_FNP(Sb, SN, Pd_one, H_one, R_one, D, TR_one, FH_one, Km, t);
     disp(i);
end

disp('The End');

%% SFR_FNP模型
function new_SFR_FNP(Sb, SN, Pd, H, R, D, TR, FH, Km, t)

% 输入函数基本参数求解

Wn = sqrt((D*R+Km)/(2*H*R*TR));
k  = (2*H*R+(D*R+Km*FH)*TR)/(2*(D*R+Km))*Wn;
Wr = Wn*sqrt(1-k^2);
g  = sqrt((1-2*TR*k*Wn+TR^2*Wn^2)/(1-k^2));
y  = atan(Wr*TR/(1-k*Wn*TR))-atan(sqrt(1-k*k)/(-k));

% 求解SFR的最大频差dfmax_SFR及其时刻tnadir_SFR

tnadir_SFR = 1/Wr*atan(Wr*TR/(k*Wn*TR-1));
dfmax_SFR  =  60*(R*Pd)/(D*R+Km)*(1+g*exp(-k*Wn*tnadir_SFR)*sin(Wr*tnadir_SFR+y));

% 求SFR模型调速器斜坡响应Pm并对其进行二项式拟合: Pm=p2*t*t+p1*t+p0

Pm = Km/R.*(+FH*TR.*(1-exp(-1/TR.*t))+t-TR.*(1-exp(-1/TR.*t)));

p = polyfit(t,Pm,2);
p0=p(3);
p1=p(2);
p2=p(1);
% plot(t,Pm,t,p2.*t.*t+p1.*t+p0)

% 求解FNP的最大频差dfmax_FNP及其时刻tnadir_FNP

tnadir_FNP = (-p1/2+sqrt(p1*p1/4+H*p2*16/3))/(p2*4/3);
dfmax_FNP  = 60*Pd*tnadir_FNP/H/4;

%求解SFR的频率响应f
t=0:0.01:1;        %仿真时间
f = 60*(R.*Pd)./(D.*R+Km).*(1+g.*exp(-k.*Wn.*t).*sin(Wr.*t+y));

data = [Pd H R D TR FH tnadir_SFR dfmax_SFR tnadir_FNP dfmax_FNP];
writematrix(data, 'sfr_fnp_data_1.csv', 'WriteMode', 'append'); %追加模式写入数据
end
