
function processAndPostImages(imageDir, processedDir, apiUrl)

    
    % Get a list of all image files in the directory
    arcFiles = dir(fullfile(imageDir, '*.jpeg'));
    processedFiles = dir(fullfile(processedDir, '*.jpeg'));
    disp('we are here');
    disp(num2str(length(arcFiles)));
    disp('we are here');
    for k = 1:length(processedFiles)
        disp(processedFiles(k).name);
    end

    % Loop over each image file
    for g = 1:length(arcFiles)
        % Read in the image
        imagePath = fullfile(arcFiles(g).folder, arcFiles(g).name);
        originalImage = imread(imagePath);

        % Perform some image manipulation (e.g., convert to grayscale)
        processedImage = rgb2gray(originalImage);

        % Save the processed image to a file with a unique name
        processedImagePath = fullfile(processedDir, 'processed.jpeg');
        imwrite(processedImage, processedImagePath);
        
        % Read the processed image file as binary data
        fileID = fopen(processedImagePath, 'rb');
        binaryData = fread(fileID, inf, 'uint8=>uint8');
        fclose(fileID);

        % Encode the binary data as a base64 string
        base64String = matlab.net.base64encode(binaryData);

        % Construct the POST data
        postData = struct('base64String', base64String);

        % Send POST request to Flask API
        % options = weboptions('MediaType', 'application/json', 'RequestMethod', 'post', 'ContentType', 'json');
        % response = webwrite(apiUrl, postData, options);

        % % Handle response
        % if ischar(response)
        %     response = jsondecode(response);
        % end

        % if isfield(response, 'message')
        %     disp(['Response from server for ', arcFiles(g).name, ': ', response.message]);
        % elseif isfield(response, 'error')
        %     disp(['Error from server for ', arcFiles(g).name, ': ', response.error]);
        % end
    end
end
