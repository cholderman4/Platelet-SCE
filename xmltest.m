clear
format long
%rng(5, 'twister')

%% Initializing nodes and parameters
griffin = 4;
phoenix = 14;


%% Initial XML stuff


docNode = com.mathworks.xml.XMLUtils.createDocument('data');

data = docNode.getDocumentElement;
% toc.setAttribute('version','2.0');

%% Settings

product = docNode.createElement('settings');
data.appendChild(product);

settingsList = {'griffin', 'phoenix'};
values = [griffin, phoenix];

for k = 1:numel(settingsList)
   curr_node = docNode.createElement(settingsList(k));
   curr_node.appendChild(docNode.createTextNode(num2str(values(k), 15)));
   product.appendChild(curr_node);
end

xmlwrite('test.xml',docNode);
type('test.xml');
