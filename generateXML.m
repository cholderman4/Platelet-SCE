clear
format long

memNodeCount = 100;
intNodeCount = 30;
fixedNodeCount = 1;
fixedNodeID = 50;


mem_r = 1.0;
mem_th = linspace(0, 2*pi, memNodeCount+1);
memNode_x = mem_r*cos(mem_th(1:end-1));
memNode_y = mem_r*sin(mem_th(1:end-1));


int_r = 0.2 * mem_r * rand(intNodeCount, 1);
int_th = 2*pi * rand(intNodeCount, 1);
intNode_x = int_r .* cos(int_th(1:end));
intNode_y = int_r .* sin(int_th(1:end));

docNode = com.mathworks.xml.XMLUtils.createDocument('data');

data = docNode.getDocumentElement;
% toc.setAttribute('version','2.0');

product = docNode.createElement('settings');
data.appendChild(product);

settingsList = {'memNodeCount', 'intNodeCount', 'viscousDamp', 'memSpringStiffness', 'memNodeMass', 'temperature', 'kB'};
values = [memNodeCount, intNodeCount, 3.769911184308, 200.0, 1.0, 300.0, 1.3806488e-8];

for k = 1:numel(settingsList)
   curr_node = docNode.createElement(settingsList(k));
   curr_node.appendChild(docNode.createTextNode(num2str(values(k), 15)));
   product.appendChild(curr_node);
end

product = docNode.createElement('membrane-nodes');
%product.setAttribute('default-mass', '1.0');
data.appendChild(product)

for i = 1:memNodeCount
    curr_node = docNode.createElement('mem-node');
    curr_node.appendChild(docNode.createTextNode(num2str([memNode_x(i), memNode_y(i), 0])));
    product.appendChild(curr_node);
end


product = docNode.createElement('interior-nodes');
%product.setAttribute('default-mass', '1.0');
data.appendChild(product)

for i = 1:intNodeCount
    curr_node = docNode.createElement('int-node');
    curr_node.appendChild(docNode.createTextNode(num2str([intNode_x(i), intNode_y(i), 0])));
    product.appendChild(curr_node);
end


product = docNode.createElement('links');
data.appendChild(product);

% Connect everything in a circle.
for j = 1:memNodeCount
   curr_node = docNode.createElement('link');
   
   node_R = j;
   if ( j == memNodeCount)
       node_R = 0;
   end
       
   curr_node.appendChild(docNode.createTextNode(num2str([j-1, node_R])));
   product.appendChild(curr_node);
end


product = docNode.createElement('fixed');
data.appendChild(product)

for i = 1:fixedNodeCount
    curr_node = docNode.createElement('nodeID');
    curr_node.appendChild(docNode.createTextNode(num2str(fixedNodeID(i))));
    product.appendChild(curr_node);
end



xmlwrite('info.xml',docNode);
type('info.xml');
