docNode = com.mathworks.xml.XMLUtils.createDocument('data');

data = docNode.getDocumentElement;
% toc.setAttribute('version','2.0');

product = docNode.createElement('nodes');
product.setAttribute('default-mass', '1.0');

data.appendChild(product)

% product.appendChild(docNode.createComment(' Functions '));

for i = 1:numel(x)
    curr_node = docNode.createElement('node');
    
    %curr_file = [functions{idx} '_help.html']; 
    %curr_node.setAttribute('target',curr_file);
    
    % Child text is the function name.
    curr_node.appendChild(docNode.createTextNode(num2str([x(i), 0, 0])));
    product.appendChild(curr_node);
end


product = docNode.createElement('links');
data.appendChild(product);

% Connect everything in a line.
for j = 1:numel(x)-1
   curr_node = docNode.createElement('link');
   curr_node.appendChild(docNode.createTextNode(num2str([j-1, j])));
   product.appendChild(curr_node);
end



xmlwrite('info.xml',docNode);
type('info.xml');
