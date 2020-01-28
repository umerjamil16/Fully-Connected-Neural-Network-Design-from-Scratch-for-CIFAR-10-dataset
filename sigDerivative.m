function g = sigDerivative(z)
    g = sigmoid(z).*(1-sigmoid(z));
end
