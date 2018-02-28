function [f,dfdx] = NN(weights,biases,input)
    %Weights and Biases must be cell arrays of Matrices/Vectors
    %Input must be a 1D Vector
    
    depth = size(weights);
    depth = depth(2);
    chain = cell(0);
    
    %FeedForward
    layer = weights{1}*input;
    dim = size(layer);
    dim = dim(1);
    dydx = zeros(dim,dim);
    z = 0;
    for i=1:dim
        if layer(i) > 0
            dydx(i,i) = 1;
        else
            layer(i)=0;
            z = 1;
        end
    end
    layer = layer + biases{1};
    
    %Append dL_1/dx evaluated at W_1*x+b_1 to chain
    %If dL_1/dx = I, append 1 instead to make it faster
    if z == 0
        chain{1} = 1;
    else
        chain{1} = dydx;
    end
    
    for j=2:depth
        layer = weights{j}*layer;
        dim = size(layer);
        dim = dim(1);
        if j ~= depth
            dydx = zeros(dim,dim);
            z = 0;
            for i=1:dim
                if layer(i) > 0
                    dydx(i,i) = 1;
                else
                    layer(i) = 0;
                    z = 1;
                end
            end
            
            %Append dL_1/dx evaluated at W_i*L_{i-1}+b_i to chain
            %If dL_1/dx = I, append 1 instead to make it faster
            if z == 0
                chain{j} = 1;
            else
                chain{j} = dydx;
            end
            
        end
        layer = layer + biases{j};
    end
    
    f = layer;
    
    %Finding the gradient
    
    df = chain{1}*weights{1};
    for i=2:depth
        df = weights{i}*df;
        if i~= depth
            df = chain{i}*df;
        end
    end
    
    dfdx = df;