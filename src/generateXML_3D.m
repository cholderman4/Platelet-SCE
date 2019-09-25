clear
format long
%rng(5, 'twister')

%% Output Choices
nIntNode = 2000;
UseMembrane = true;
nMemNode = 0;

NodeFileName = 'nodeInfo.xml';
ParameterFileName = 'paramInfo.xml';






%% Independent Parameters

    U_II = 1.0;
    P_II = 2.15;

    viscousDamp = 3.769911184308;
    springStiffness_MM = 500.0;
    springStiffness_MP = 600.0;
    memNodeMass = 1.0;
    temperature = 300.0;
    kB = 1.3806488e-8;
    
%% Parameter Dependencies

% Morse parameters
R_cell = 1.0;
d = 3;
density = 0.640;
intNodeScale = 0.950;



%% Dependent Parameters

% Morse parameters
rEq_II = intNodeScale * R_cell *(density / nIntNode)^(1/d); 
U_MI = 20*U_II;
P_MI = P_II;
rEq_MI = rEq_II;


%% Initializing nodes and parameters


meshFaceCount = 0;
if (UseMembrane)
    mesh = icoSphereMesh(4);
    
    nMemNode = size(mesh.x,1);
    meshFaceCount = size(mesh.face, 1);
    
    memNode_x = R_cell * mesh.x;
    memNode_y = R_cell * mesh.y;
    memNode_z = R_cell * mesh.z;    
end

fixedNodeCount = 0;
fixedNodeID = 50;



int_r = intNodeScale * R_cell * rand(nIntNode, 1);
int_th = 2*pi * rand(nIntNode, 1);
int_phi = pi * rand(nIntNode, 1); % Set to pi for 2D.

intNode_x = int_r .* cos(int_th) .* sin(int_phi);
intNode_y = int_r .* sin(int_th) .* sin(int_phi);
intNode_z = int_r .* cos(int_phi);

%% Parameter List

    n={}; v=[];

    n = [n 'viscousDamp'];          v=[v viscousDamp];
    n = [n 'temperature'];          v=[v temperature];
    n = [n 'kB'];                   v=[v kB];
    n = [n 'springStiffness_MM'];   v=[v springStiffness_MM];
    n = [n 'springStiffness_MP'];   v=[v springStiffness_MP];
    n = [n 'U_II'];                 v=[v U_II];
    n = [n 'P_II'];                 v=[v P_II];
    n = [n 'rEq_II'];               v=[v rEq_II];
    n = [n 'U_MI'];                 v=[v U_MI];
    n = [n 'P_MI'];                 v=[v P_MI];
    n = [n 'rEq_MI'];               v=[v rEq_MI];
    n = [n 'nMemNode'];         v=[v nMemNode];
    n = [n 'nIntNode'];         v=[v nIntNode];


%% Initial XML stuff


docNode = com.mathworks.xml.XMLUtils.createDocument('data');

data = docNode.getDocumentElement;
% toc.setAttribute('version','2.0');

%% Parameters

docParam = com.mathworks.xml.XMLUtils.createDocument('data');

param = docParam.getDocumentElement;
product = docParam.createElement('parameters');
param.appendChild(product);

for k = 1:numel(n)
   curr_node = docParam.createElement(n(k));
   curr_node.appendChild(docParam.createTextNode(num2str(v(k), 15)));
   product.appendChild(curr_node);
end


xmlwrite(ParameterFileName,docParam);


%% Membrane Nodes

product = docNode.createElement('membrane-nodes');
%product.setAttribute('default-mass', '1.0');
data.appendChild(product);

for i = 1:nMemNode
    curr_node = docNode.createElement('mem-node');
    curr_node.appendChild(docNode.createTextNode(num2str([memNode_x(i), memNode_y(i), memNode_z(i)])));
    product.appendChild(curr_node);
end


%% Interior Nodes


product = docNode.createElement('interior-nodes');
%product.setAttribute('default-mass', '1.0');
data.appendChild(product);

for i = 1:nIntNode
    curr_node = docNode.createElement('int-node');
    curr_node.appendChild(docNode.createTextNode(num2str([intNode_x(i), intNode_y(i), intNode_z(i)])));
    product.appendChild(curr_node);
end


%% Membrane Spring Links


product = docNode.createElement('links');
data.appendChild(product);

for j = 1:meshFaceCount
    for k = 1:3
       curr_node = docNode.createElement('link');

       node_R = k+1;
       if ( k == 3)
           node_R = 1;
       end

       curr_node.appendChild(docNode.createTextNode(num2str([mesh.face(j, k)-1, mesh.face(j, node_R)-1])));
       product.appendChild(curr_node);
    end
end


%% Fixed Nodes



product = docNode.createElement('fixed');
data.appendChild(product);

for i = 1:fixedNodeCount
    curr_node = docNode.createElement('nodeID');
    curr_node.appendChild(docNode.createTextNode(num2str(fixedNodeID(i))));
    product.appendChild(curr_node);
end



xmlwrite(NodeFileName,docNode);
%type('info.xml');
