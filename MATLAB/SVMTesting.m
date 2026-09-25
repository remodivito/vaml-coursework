function [prediction maxi]= SVMTesting(image,model)

if strcmp(model.type,'binary')
    pred = svmval(image,model.xsup,model.w,model.w0,model.param.kernel,model.param.kerneloption)
    
    if pred>0.7
        prediction = 1;
    else
        prediction = -1;
    end
    
    maxi=pred;
    
end