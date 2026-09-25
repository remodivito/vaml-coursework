function [selectedObjects, removedObjects] = NMSwithOverlapScores(Objects, threshold)
    % Sort Objects based on confidence scores in descending order
    [~, order] = sort([Objects.confidence], 'descend');
    Objects = Objects(order);

    selectedObjects = []; % Array to hold selected (retained) objects
    removedObjects = [];  % Array to hold removed objects for inspection

    i = 1;
    while i <= numel(Objects)
        % Get the bounding box for the current object
        bbox1 = Objects(i).bbox;
        
        % Add the current object to the selected list
        selectedObjects = [selectedObjects; Objects(i)]; %#ok<AGROW>

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
                % Add the removed object to the removedObjects list
                if Objects(i).confidence > Objects(j).confidence
                    removedObjects = [removedObjects; Objects(j)]; %#ok<AGROW>
                    Objects(j) = []; % Remove j-th object
                else
                    removedObjects = [removedObjects; Objects(i)]; %#ok<AGROW>
                    selectedObjects(end) = []; % Remove i-th object from selected list
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
