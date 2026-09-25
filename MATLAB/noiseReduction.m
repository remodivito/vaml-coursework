function Iout = noiseReduction(I,N)
mask = ones(N);
mask = mask/sum(sum(mask));

Iout = filter2(mask, I);
end