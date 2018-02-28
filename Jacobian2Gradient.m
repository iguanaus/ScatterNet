function gradient = Jacobian2Gradient(dfdx,out,expectedOut)
    gradient = transpose(dfdx)*(out-expectedOut);
end