function W = initW(inputLayer, outputLayer)
    ep_pos = 0.1;
    ep_neg = -0.1;
    W = (ep_neg-ep_pos).*rand(outputLayer, inputLayer + 1) + ep_pos;
end
