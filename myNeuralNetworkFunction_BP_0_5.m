function [Y,Xf,Af] = myNeuralNetworkFunction(X,~,~)
%MYNEURALNETWORKFUNCTION neural network simulation function.
%
% Auto-generated by MATLAB, 06-Aug-2024 15:26:32.
%
% [Y] = myNeuralNetworkFunction(X,~,~) takes these arguments:
%
%   X = 1xTS cell, 1 inputs over TS timesteps
%   Each X{1,ts} = Qx4 matrix, input #1 at timestep ts.
%
% and returns:
%   Y = 1xTS cell of 1 outputs over TS timesteps.
%   Each Y{1,ts} = Qx1 matrix, output #1 at timestep ts.
%
% where Q is number of samples (or series) and TS is the number of timesteps.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [0;0;0;0];
x1_step1.gain = [0.112959455080705;0.112428641585882;0.149534797819244;0.121531796790557];
x1_step1.ymin = -1;

% Layer 1
b1 = [2.2660596223111655334;1.5714982999742237446;-1.4429101864980224956;-0.7899946611144318398;0.27220216193765051083;-0.064864199264594385452;1.0125524184596577104;1.8178971803652137496;-1.8696462357748488969;1.8445264706669504129];
IW1_1 = [-1.5886809492241305009 -1.1309632070933417491 1.1360211174225296471 1.2956967522811806415;-0.74714509857270694493 -0.32228132623105643084 1.4140975407944920139 0.87461859273337072551;0.53652303665593170656 2.281117718880633749 -0.53071091091936095641 -0.7675510518705308538;1.5740702649516866707 0.99280906860192552621 0.50734813841930681555 1.5524548722596831496;0.70912967731050036146 0.819131822535479448 -0.15823721077593208562 -0.889073693293102707;-1.9929362782803312637 0.69124424453375954425 1.1581939349690584251 -0.57953652815044465196;1.2541403766730352398 -1.5368271173461676149 -1.4870914005877418074 -0.51259697235909340574;0.89269742135787122361 0.5816433937087073236 0.10680411766382601202 0.3605557453551178626;-0.036914195840674478288 0.71077307273806700216 -1.9236070846704294013 0.71012528471062363877;1.5781234817747209487 0.83295097170348209037 -1.6168519565299801499 -0.5413561307910199627];

% Layer 2
b2 = 0.094735076671184803576;
LW2_1 = [-0.51173036405649119374 0.39639379196801333149 0.18865062627676373874 0.38447953137028789694 0.4179110828064795391 0.081634144854367249322 -0.06767847242163013699 0.97406535310510733439 -0.43990970622105640686 -0.33279739877624842093];

% Output 1
y1_step1.ymin = -1;
y1_step1.gain = 0.0816778506167554;
y1_step1.xoffset = 0;

% ===== SIMULATION ========

% Format Input Arguments
isCellX = iscell(X);
if ~isCellX
    X = {X};
end

% Dimensions
TS = size(X,2); % timesteps
if ~isempty(X)
    Q = size(X{1},1); % samples/series
else
    Q = 0;
end

% Allocate Outputs
Y = cell(1,TS);

% Time loop
for ts=1:TS
    
    % Input 1
    X{1,ts} = X{1,ts}';
    Xp1 = mapminmax_apply(X{1,ts},x1_step1);
    
    % Layer 1
    a1 = tansig_apply(repmat(b1,1,Q) + IW1_1*Xp1);
    
    % Layer 2
    a2 = repmat(b2,1,Q) + LW2_1*a1;
    
    % Output 1
    Y{1,ts} = mapminmax_reverse(a2,y1_step1);
    Y{1,ts} = Y{1,ts}';
end

% Final Delay States
Xf = cell(1,0);
Af = cell(2,0);

% Format Output Arguments
if ~isCellX
    Y = cell2mat(Y);
end
end

% ===== MODULE FUNCTIONS ========

% Map Minimum and Maximum Input Processing Function
function y = mapminmax_apply(x,settings)
y = bsxfun(@minus,x,settings.xoffset);
y = bsxfun(@times,y,settings.gain);
y = bsxfun(@plus,y,settings.ymin);
end

% Sigmoid Symmetric Transfer Function
function a = tansig_apply(n,~)
a = 2 ./ (1 + exp(-2*n)) - 1;
end

% Map Minimum and Maximum Output Reverse-Processing Function
function x = mapminmax_reverse(y,settings)
x = bsxfun(@minus,y,settings.ymin);
x = bsxfun(@rdivide,x,settings.gain);
x = bsxfun(@plus,x,settings.xoffset);
end
