function Objects = NMS(Objects, threshold)
    % Objects is an Nx4 matrix, where each row represents [x, y, width, height]
    % along with an Nx1 vector for confidence scores (e.g., Objects(:,5))
    
    % Sort Objects based on confidence scores in descending order
    [~, order] = sort([Objects.confidence], 'descend');
    Objects = Objects(order);

    i = 1;
    while i <= numel(Objects)
        % Get the bounding box for the current object
        bbox1 = Objects(i).bbox; % Assuming Objects(i).bbox = [x, y, width, height]
        
        j = i + 1;
        while j <= numel(Objects)
            % Get the bounding box for the next object
            bbox2 = Objects(j).bbox;

            % Calculate the intersection area
            intersection_area = rectint(bbox1, bbox2);
            
            % Calculate area of the bounding box of the first object
            bbox1_area = bbox1(3) * bbox1(4);
            
            % Check the overlap threshold
            if (intersection_area / bbox1_area) > threshold
                % Remove the one with the smaller confidence score
                if Objects(i).confidence > Objects(j).confidence
                    Objects(j) = []; % Remove j-th object
                else
                    Objects(i) = []; % Remove i-th object and break to avoid incrementing i
                    break;
                end
            else
                j = j + 1; % Move to the next object
            end
        end
        i = i + 1;
    end
end
