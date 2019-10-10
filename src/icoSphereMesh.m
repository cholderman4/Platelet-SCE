function mesh = icoSphereMesh(n)
    % Copyright (c) 2015, Matthew Kelly
    % All rights reserved.
    
    % Redistribution and use in source and binary forms, with or without
    % modification, are permitted provided that the following conditions are met:
    
    % * Redistributions of source code must retain the above copyright notice, this
    %   list of conditions and the following disclaimer.
    
    % * Redistributions in binary form must reproduce the above copyright notice,
    %   this list of conditions and the following disclaimer in the documentation
    %   and/or other materials provided with the distribution
    % THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    % AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    % IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    % DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
    % FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    % DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    % SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    % CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    % OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    % OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
% mesh = icoSphereMesh(n)
%
% This code returns a triangle mesh for the unit icosphere in 3 dimensions
%
% INPUTS:
%   n = recursion level:    (default = 1)  
%          n == 0   returns 12 verticies
%          n == 1   returns 42 verticies
%          n == 2   returns 162 verticies
%          n == 3   returns 642 verticies
%          n == 4   returns 2562 verticies
%          n == 5   returns 10242 verticies
%          n > 5    set n == 5 to avoid huge mesh. 
%
% OUTPUTS:
%   mesh = struct with fields:
%   mesh.face = [M x 3] array of indicies for each triangle
%   mesh.x = the x-coordinate of each vertex
%   mesh.y = the y-coordinate of each vertex
%   mesh.z = the z-coordinate of each vertex
%
% NOTES:
%  1) Plot using:  trimesh(mesh.face, mesh.x, mesh.y, mesh.z);
%
%  2) My code is based on code from two online sources:
%     http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html
%     http://eeg.sourceforge.net/doc_m2html/bioelectromagnetism/mesh_refine_tri4.html
%

if nargin==0
   n = 1; 
end
if n<0, n=0; elseif n>5, n=5; end

t = 0.5*(1 + sqrt(5));  %The golden ratio!

vertex = projectToSphere([...
    -1  t  0;
    1  t  0;
    -1 -t  0;
    1 -t  0;
    0 -1  t;
    0  1  t;
    0 -1 -t;
    0  1 -t;
    t  0 -1;
    t  0  1;
    -t  0 -1;
    -t  0  1]);

face = [...
    1    12     6;
    1     6     2;
    1     2     8;
    1     8    11;
    1    11    12;
    2     6    10;
    6    12     5;
    12    11     3;
    11     8     7;
    8     2     9;
    4    10     5;
    4     5     3;
    4     3     7;
    4     7     9;
    4     9    10;
    5    10     6;
    3     5    12;
    7     3    11;
    9     7     8;
    10     9     2];

for i=1:n
    [face, vertex] = refineSphere(face, vertex);
end
mesh.face = face;
mesh.x = vertex(:,1);
mesh.y = vertex(:,2);
mesh.z = vertex(:,3);

end

function v = projectToSphere(v)
len = sqrt(sum(v.^2,2))*ones(1,3);
v = v./len;
end

function [face, vertex] = refineSphere(f, v)

nFace = size(f,1);
nVertex = size(v,1);

face = zeros(4*nFace,3);
vertex = zeros(3*nFace,3);

vertex(1:nVertex,:) = v;  %Initialize new verticies
n = nVertex;

for i=1:nFace
    
    %Get the index for each verted in the original mesh:
    iA = f(i,1);
    iB = f(i,2);
    iC = f(i,3);
    
    %Get the coordinates of each vertex
    A = v(iA,:); B = v(iB,:); C = v(iC,:);
    
    %Get the midpoints of each edge:
    a = 0.5*(A+B); b = 0.5*(B+C); c = 0.5*(A+C);
    
    %Find the new indicies of these vertices
    [ia,vertex,n] = addVertex(a,vertex,n);
    [ib,vertex,n] = addVertex(b,vertex,n);
    [ic,vertex,n] = addVertex(c,vertex,n);
    
    %Store the new triangles:
    face(i*4-0,:) = [iA,ia,ic];
    face(i*4-1,:) = [ia,iB,ib];
    face(i*4-2,:) = [ic,ib,iC];
    face(i*4-3,:) = [ia,ib,ic];
    
end

vertex = vertex(1:n,:);
vertex = projectToSphere(vertex);

end

function [idx,v,n] = addVertex(p,v,n)
% p = text vertex
% v = list of verticies
% n = number of non-zero entries in v

match = v(1:n,1)==p(1) & v(1:n,2)==p(2) & v(1:n,3)==p(3);
idx = find(match);

if isempty(idx)   %Then create a new point!
   n=n+1;
   v(n,:) = p;
   idx = n;
elseif length(idx)>1
    error('Somehow we have a redundant point!');
end


end




