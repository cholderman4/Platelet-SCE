function printParametersToXML(...
    a,...
    b,...
    c)
    
    writeParameterstoXML(...
        a,...
        b,...
        c);
    
end


function writeParameterstoXML(...
    a,...
    b,...
    c)
    
    docNode = com.mathworks.xml.XMLUtils.createDocument('parameters');
    data = docNode.getDocumentElement;

    for m = 1:nargin
        disp(['Calling variable ' num2str(m) ' is ''' inputname(m) '''.'])
    end


% product = docNode.createElement('settings');
data.appendChild(docNode);

% settingsList = {'morse-U_MI', 'morse-P_MI', 'morse-R_eq_MI', 'morse-U_II', 'morse-P_II', 'morse-R_eq_II', 'memNodeCount', 'intNodeCount', 'viscousDamp', 'memSpringStiffness', 'memNodeMass', 'temperature', 'kB'};
% values = [U_MI, P_MI, R_eq_MI, U_II, P_II, R_eq_II, memNodeCount, intNodeCount, 3.769911184308, 500.0, 1.0, 300.0, 1.3806488e-8];

    for k = 1:numel(settingsList)
       curr_node = docNode.createElement(inputname(m));
       curr_node.appendChild(docNode.createTextNode(num2str(values(k), 15)));
       data.appendChild(curr_node);
    end


xmlwrite('parameters.xml',docNode);




end

