clear
format long
%rng(5, 'twister')

%% Initializing nodes and parameters

UseMembrane = true;
memNodeCount = 0;
meshFaceCount = 0;

mesh = icoSphereMesh(4);

if (UseMembrane)
    memNodeCount = size(mesh.x,1);
    meshFaceCount = size(mesh.face, 1);
end

intNodeCount = 2000;
fixedNodeCount = 0;
fixedNodeID = 50;

intNodeScale = 0.950;

% Morse parameters
R_cell = 1.0;
U_II = 1.0;
P_II = 2.15;
d = 3;
density = 0.640;
R_eq_II = intNodeScale * R_cell *(density / intNodeCount)^(1/d); 
%R_eq = 0.15;

U_MI = 20*U_II;
P_MI = P_II;
R_eq_MI = R_eq_II;


memNode_x = R_cell * mesh.x;
memNode_y = R_cell * mesh.y;
memNode_z = R_cell * mesh.z;



% mem_th = linspace(0, 2*pi, memNodeCount+1);
% memNode_x = R_cell * cos(mem_th(1:end-1));
% memNode_y = R_cell * sin(mem_th(1:end-1));



int_r = intNodeScale * R_cell * rand(intNodeCount, 1);
int_th = 2*pi * rand(intNodeCount, 1);
int_phi = pi * rand(intNodeCount, 1); % Set to pi for 2D.

intNode_x = int_r .* cos(int_th) .* sin(int_phi);
intNode_y = int_r .* sin(int_th) .* sin(int_phi);
intNode_z = int_r .* cos(int_phi);

%% Initial XML stuff


docNode = com.mathworks.xml.XMLUtils.createDocument('data');

data = docNode.getDocumentElement;
% toc.setAttribute('version','2.0');

%% Settings

product = docNode.createElement('settings');
data.appendChild(product);

settingsList = {'morse-U_MI', 'morse-P_MI', 'morse-R_eq_MI', 'morse-U_II', 'morse-P_II', 'morse-R_eq_II', 'memNodeCount', 'intNodeCount', 'viscousDamp', 'memSpringStiffness', 'memNodeMass', 'temperature', 'kB'};
values = [U_MI, P_MI, R_eq_MI, U_II, P_II, R_eq_II, memNodeCount, intNodeCount, 3.769911184308, 500.0, 1.0, 300.0, 1.3806488e-8];

for k = 1:numel(settingsList)
   curr_node = docNode.createElement(settingsList(k));
   curr_node.appendChild(docNode.createTextNode(num2str(values(k), 15)));
   product.appendChild(curr_node);
end

%% Membrane Nodes

product = docNode.createElement('membrane-nodes');
%product.setAttribute('default-mass', '1.0');
data.appendChild(product);

for i = 1:memNodeCount
    curr_node = docNode.createElement('mem-node');
    curr_node.appendChild(docNode.createTextNode(num2str([memNode_x(i), memNode_y(i), memNode_z(i)])));
    product.appendChild(curr_node);
end


%% Interior Nodes


product = docNode.createElement('interior-nodes');
%product.setAttribute('default-mass', '1.0');
data.appendChild(product);

for i = 1:intNodeCount
    curr_node = docNode.createElement('int-node');
    curr_node.appendChild(docNode.createTextNode(num2str([intNode_x(i), intNode_y(i), intNode_z(i)])));
    product.appendChild(curr_node);
end


%% Membrane Spring Links


product = docNode.createElement('links');
data.appendChild(product);

% Connect everything in a circle.
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



xmlwrite('info.xml',docNode);
%type('info.xml');
