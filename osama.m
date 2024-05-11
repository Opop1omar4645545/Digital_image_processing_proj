classdef osama < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure               matlab.ui.Figure
        UploadImageButton      matlab.ui.control.Button
        ResetButton            matlab.ui.control.Button
        FeaturesDropDownLabel  matlab.ui.control.Label
        FeaturesDropDown       matlab.ui.control.DropDown
        Input1EditFieldLabel   matlab.ui.control.Label
        Input1EditField        matlab.ui.control.NumericEditField
        Input2EditFieldLabel   matlab.ui.control.Label
        Input2EditField        matlab.ui.control.NumericEditField
        UIAxes                 matlab.ui.control.UIAxes
        UIAxes_2               matlab.ui.control.UIAxes
        ApplyButton            matlab.ui.control.Button
        Image                  matlab.ui.control.Image
        RGB2GrayButton         matlab.ui.control.Button
    end

    % Callbacks that handle component events
    methods (Access = private)

        % Button pushed function: UploadImageButton
        function UploadImageButtonPushed(app, event)
        global image;
        global arr;
        global counter;
        try
            [filename1,filepath1] = uigetfile({'*.*'},'Select Data File 1');
            cd(filepath1);
            image=imread([filepath1 filename1]);
            counter=1;
            arr = cell(1,100);
            app.UIAxes.Visible = 'on';
            imshow(image, 'Parent', app.UIAxes);
            arr{counter}=image;
        catch
            uialert(app.UIFigure,["No Image has been selected"; ...
                            "Please select an Image"],'Invalid choice');
        end
        clear uploadedImage;
        app.UIAxes_2.Visible = 'off';
        end

        % Button pushed function: ApplyButton
        function ApplyButtonPushed(app, event)
            global arr;
            global counter;
            global img;
            Feature=app.FeaturesDropDown.Value;
            if(counter>1)
                
            end
            switch Feature
                case "Sampling Down" %1
                    originalImage = arr{counter};
                    [rows, cols, matricesNo] = size(originalImage);
                    SamplingFactor = app.Input1EditField.Value;
                    for metricesIndex=1:1:matricesNo
                       resizedImage(:,:,metricesIndex) = subSampling(originalImage(:,:,metricesIndex),SamplingFactor);
                    end
                    imshow(resizedImage, 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=resizedImage;
                case "Resampling Up" %2
                    originalImage = arr{counter};
                    [rows, cols, matricesNo] = size(originalImage);
                    SamplingFactor = app.Input1EditField.Value;
                    for metricesIndex=1:1:matricesNo
                        resizedImage(:,:,metricesIndex) = upSampling(originalImage(:,:,metricesIndex),SamplingFactor);
                    end
                    imshow(resizedImage, 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=resizedImage;
                case "Gray Scale" %4
                    k = app.Input1EditField.Value;
                    target_levels = 2^k;
                    target_compr_factor = 256 / target_levels;
                    reduced_image = uint8(floor(double(arr{counter})/256 * target_levels) * target_compr_factor);
                    imshow(reduced_image, 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=reduced_image;
                case "Identity(linear)" %5
                    id=arr{counter};
                    imshow(id, 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=id;
                case "Negative(linear)" %6
                    neg = 255 - arr{counter};
                    imshow(neg, 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=neg;
                case "Log Trans" %7
                    c=app.Input1EditField.Value;
                    doubleImage=im2double(arr{counter});
                    s=(c*log(1+doubleImage))*256;
                    s1=uint8(s);
                    imshow(s1, 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=s1;
                case "Inv Log" %8
                    L=app.Input1EditField.Value;
                    doubleImage=arr{counter};
                    exp_I = uint8((exp(double(doubleImage)) .^ (log(L) / (L-1))) - 1);
                    imshow(exp_I, 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=exp_I;
                case "Power Trans" %9
                    c = app.Input1EditField.Value;
                    p = app.Input2EditField.Value;
                    x1 = double(arr{counter});
                    y = c * power(x1, p);
                    output_img = uint8(y);
                    imshow(output_img, 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=output_img;
                case "Contrast Stretch" %10
                    originalImage = arr{counter};
                    rm= min(originalImage,[],1);
                    rM= max(originalImage,[],1);
                    sm= app.Input1EditField.Value;
                    sM= app.Input2EditField.Value;
                    s1= double(((sM-sm) ./ (rM-rm)) .* (originalImage - rm) + sm);
                    s = uint8(s1);
                    imshow(s, 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=s;
                case "Threasholding" %11
                    originalImage = arr{counter};    
                    newImage = originalImage;
                    T= app.Input1EditField.Value;
                    [rows, cols] = size(originalImage);
                    for row_index=1:1:rows
                        for col_index=1:1:cols
                            if(originalImage(row_index,col_index) >= T)
                                newImage(row_index,col_index) = 255;
                            else
                                 newImage(row_index,col_index) = 0;
                            end
                        end
                    end
                    imshow(newImage, 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=newImage;
                case "Gray Scale AP#1" %12
                    originalImage = arr{counter}; 
                    newImage = originalImage;
                    sm= app.Input1EditField.Value;
                    sM= app.Input2EditField.Value;
                    [rows, cols] = size(originalImage);
                    for row_index=1:1:rows
                        for col_index=1:1:cols
                            if(originalImage(row_index,col_index)>=sm && originalImage(row_index,col_index)<=sM)
                                newImage(row_index,col_index) = 255;
                            else
                                 newImage(row_index,col_index) = 0;
                            end
                        end
                    end
                    imshow(newImage, 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=newImage;
                case "Gray Scale AP#2" %13
                    originalImage = arr{counter}; 
                    newImage = originalImage;
                    sm= app.Input1EditField.Value;
                    sM= app.Input2EditField.Value;
                    [rows, cols] = size(originalImage);
                    for row_index=1:1:rows
                        for col_index=1:1:cols
                            if(originalImage(row_index,col_index)>=sm && originalImage(row_index,col_index)<=sM)
                                newImage(row_index,col_index) = 255;
                            else
                                 newImage(row_index,col_index) = originalImage(row_index,col_index);
                            end
                        end
                    end
                    imshow(newImage, 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=newImage;
                case "Bit-Plane Slicing" %14
                    originalImage = arr{counter};
                    k= app.Input1EditField.Value;
                    [rows ,cols] = size(originalImage);
                    newImage = zeros(rows,cols,8);
                    for row_index=1:1:rows
                        for col_index=1:1:cols
                            newImage(row_index,col_index,k)=bitget(originalImage(row_index,col_index),k);
                        end
                    end
                    imshow(newImage, 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=newImage;
                case "Addition" %15
                    Image1 = arr{counter};
                    ad = pp.Input1EditField.Value;
                    answer = Image1 + ad;
                    imshow(answer,'Parent',app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=answer;
                case "Subtraction" %16
                    % yes or no question
%                     try %if yes
%                         app.Input1EditField.Visible = 'on';
%                         app.Input1EditFieldLabel.Visible = 'on';
%                     catch %if no
%                         
%                     end
                    [filename1,filepath1] = uigetfile({'.'},'Select Data File 1');
                    cd(filepath1);
                    e=imread([filepath1 filename1]);
                    Image1 = arr{counter};
                    try
                        Image2 = rgb2gray(e);
                    catch
                    end
                    f=imresize(Image1,[size(Image2,1) size(Image2,2)]);
                    answer = f-Image2;
                    a=answer;
                    imshow(a,'Parent',app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=a;
                case "Logic AND" %17
                    a=arr{counter};
                    [m, n] = size(a);
                    and_image = zeros(m, n, 'uint8');
                    for i = 1:m
                        for j = 1:n
                        and_image(i,j) = bitand( a(i,j), img(i,j));
                        end
                    end
                    a=and_image;
                    imshow(a,'Parent',app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=a;
                case "Logic OR" %18
                    
                case "Histogram" %19
                    
                case "Smoothing(Avg Standerd)" %20
                    
                case "Smoothing(Avg Weight)" %21
                    
                case "Smoothing(Median)" %22
                    
                case "Sharpining(Laplacian)" %23
                    
                case "Sharpining(Comp Laplacian)" %24
                    
                case "Sharpining(Diag Laplacian)"
                    
                case "Sharpining(Robert Vertical)" %25
                    input_image = arr{counter};
                    
                    % Convert the image to double
                    input_image = double(input_image);
                      
                    % Pre-allocate the filtered_image matrix with zeros
                    filtered_image = zeros(size(input_image));
                      
                    % Robert Operator Mask
                    My = [0 -1;1 0];
                    
                    for i = 1:size(input_image, 1) - 1
                        for j = 1:size(input_image, 2) - 1
                            
                            % Gradient approximations
                            Gy = sum(sum(My.*input_image(i:i+1, j:j+1)));
                            
                            % Calculate magnitude of vector
                            filtered_image(i, j) = sqrt(Gy.^2); 
                        end
                    end
                      
                    % Displaying Filtered Image
                    filtered_image = uint8(filtered_image);
                    imshow(filtered_image, 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=filtered_image;
                case "Sharpining(Robert Horizontal)"
                    input_image = arr{counter};
                      
                    % Convert the image to double
                    input_image = double(input_image);
                      
                    % Pre-allocate the filtered_image matrix with zeros
                    filtered_image = zeros(size(input_image));
                      
                    % Robert Operator Mask
                    Mx = [1 0; 0 -1];
                    
                    for i = 1:size(input_image, 1) - 1
                        for j = 1:size(input_image, 2) - 1
                            
                            % Gradient approximations
                            Gx = sum(sum(Mx.*input_image(i:i+1, j:j+1)));

                            % Calculate magnitude of vector
                            filtered_image(i, j) = sqrt(Gx.^2);     
                        end
                    end    
                    % Displaying Filtered Image
                    filtered_image = uint8(filtered_image);
                    imshow(filtered_image, 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=filtered_image;
                case "Sharpining(Robert ALL)"
                    input_image = arr{counter};
                      
                    % Convert the image to double
                    input_image = double(input_image);
                      
                    % Pre-allocate the filtered_image matrix with zeros
                    filtered_image = zeros(size(input_image));
                      
                    % Robert Operator Mask
                    Mx = [1 0; 0 -1];
                    My = [0 1; -1 0];
                    
                    for i = 1:size(input_image, 1) - 1
                        for j = 1:size(input_image, 2) - 1
                            
                            % Gradient approximations
                            Gx = sum(sum(Mx.*input_image(i:i+1, j:j+1)));
                            Gy = sum(sum(My.*input_image(i:i+1, j:j+1)));
                            
                            % Calculate magnitude of vector
                            filtered_image(i, j) = sqrt(Gx.^2 + Gy.^2);
                             
                        end
                    end
                    % Displaying Filtered Image
                    filtered_image = uint8(filtered_image);
                    imshow(filtered_image, 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=filtered_image;
                case "Sharpining(Sobel Vertical)"
                    I=double(arr{counter});
                    In=I;
                    mask2 = [-1 0 1;-2 0 2;-1 0 1];
                    
                    for i=2:size(I, 1)-1
                        for j=2:size(I, 2)-1
                            neighbour_matrix2=mask2.*In(i-1:i+1, j-1:j+1);
                            avg_value2=sum(neighbour_matrix2(:));

                            %using max function for detection of final edges
                            I(i, j)=avg_value2;
                        end
                    end
                    I = uint8(I);
                    imshow(I, 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=I;
                case "Sharpining(Sobel Horizontal)"
                    I=double(arr{counter});
                    In=I;
                    mask1 = [-1 -2 -1;0 0 0;1 2 1];
                    
                    for i=2:size(I, 1)-1
                        for j=2:size(I, 2)-1
                            neighbour_matrix1=mask1.*In(i-1:i+1, j-1:j+1);
                            avg_value1=sum(neighbour_matrix1(:));

                            %using max function for detection of final edges
                            I(i, j)=avg_value1;
                        end
                    end
                    I = uint8(I);
                    imshow(I, 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=I;
                case "Sharpining(Sobel ALL)" %26
                    I=double(arr{counter});
                    In=I;
                    mask1 = [-1 -2 -1;0 0 0;1 2 1];
                    mask2 = [-1 0 1;-2 0 2;-1 0 1];
                    
                    for i=2:size(I, 1)-1
                        for j=2:size(I, 2)-1
                            neighbour_matrix1=mask1.*In(i-1:i+1, j-1:j+1);
                            avg_value1=sum(neighbour_matrix1(:));
                    
                            neighbour_matrix2=mask2.*In(i-1:i+1, j-1:j+1);
                            avg_value2=sum(neighbour_matrix2(:));
                    
                            %using max function for detection of final edges
                            I(i, j)=max([avg_value1, avg_value2]);
                    
                        end
                    end
                    I = uint8(I);
                    imshow(I, 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=I;
                case "High Pass(Ideal)" %27
                    input_image = arr{counter};  
                    [M, N] = size(input_image);
                    FT_img = fft2(double(input_image));
                    D0 = app.Input1EditField.Value; %inp
                    
                    % Designing filter
                    u = 0:(M-1);
                    idx = find(u>M/2);
                    u(idx) = u(idx)-M;
                    v = 0:(N-1);
                    idy = find(v>N/2);
                    v(idy) = v(idy)-N;

                    [V, U] = meshgrid(v, u);
                      
                    % Calculating Euclidean Distance
                    D = sqrt(U.^2+V.^2);
                      
                    % Comparing with the cut-off frequency and 
                    % determining the filtering mask
                    H = double(D >= D0);
                    
                    G = H.*FT_img;
                    
                    output_image = real(ifft2(double(G)));
                    imshow(output_image,[ ], 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=output_image;
                case "High Pass(Butterworth)" %28
                    input_image = arr{counter};  
                    [M, N] = size(input_image);
                    FT_img = fft2(double(input_image));
                    D0 = app.Input1EditField.Value; 
                    num= app.Input2EditField.Value;
                    n = num * num;
                    % Designing filter
                    u = 0:(M-1);
                    idx = find(u>M/2);
                    u(idx) = u(idx)-M;
                    v = 0:(N-1);
                    idy = find(v>N/2);
                    v(idy) = v(idy)-N;
                    
                    [V, U] = meshgrid(v, u);
                    
                    D = sqrt(U.^2+V.^2);
                    
                    D = D0./ D;
                    
                    H = 1./((1+D).^n);
                      
                    G = H.*FT_img;
                       
                    output_image = real(ifft2(double(G)));
                    imshow(output_image,[ ], 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=output_image;
                case "High Pass(Gaussian)" %29
                    input_image = arr{counter};  
                    [M, N] = size(input_image);
                    FT_img = fft2(double(input_image));
                    D0 = app.Input1EditField.Value;  
                    D0 = (D0^2)*2;
                    % Designing filter
                    u = 0:(M-1);
                    idx = find(u>M/2);
                    u(idx) = u(idx)-M;
                    v = 0:(N-1);
                    idy = find(v>N/2);
                    v(idy) = v(idy)-N;
                    
                    [V, U] = meshgrid(v, u);
                    
                    D = sqrt(U.^2+V.^2);
                    
                    D = -D.^2;
                    
                    H = 1-exp(D/D0);
                    
                    G = H.*FT_img;
                      
                    output_image = real(ifft2(double(G)));
                    imshow(output_image,[ ], 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=output_image;
                case "Low Pass(Ideal)" %30
                    input_image = arr{counter};  
                    [M, N] = size(input_image);
                    FT_img = fft2(double(input_image));
                    D0 = app.Input1EditField.Value; %inp
                    
                    % Designing filter
                    u = 0:(M-1);
                    idx = find(u>M/2);
                    u(idx) = u(idx)-M;
                    v = 0:(N-1);
                    idy = find(v>N/2);
                    v(idy) = v(idy)-N;
                    
                    [V, U] = meshgrid(v, u);
                      
                    % Calculating Euclidean Distance
                    D = sqrt(U.^2+V.^2);
                    
                    H = double(D <= D0);
                    
                    G = H.*FT_img;
                    
                    output_image = real(ifft2(double(G)));
                    imshow(output_image,[ ], 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=output_image;
                case "Low Pass(Butterworth)" %31
                    input_image = arr{counter};  
                    [M, N] = size(input_image);
                    FT_img = fft2(double(input_image));
                    D0 = app.Input1EditField.Value; 
                    num= app.Input2EditField.Value;
                    n = num * num;
                    % Designing filter
                    u = 0:(M-1);
                    idx = find(u>M/2);
                    u(idx) = u(idx)-M;
                    v = 0:(N-1);
                    idy = find(v>N/2);
                    v(idy) = v(idy)-N;
                    
                    [V, U] = meshgrid(v, u);
                    
                    D = sqrt(U.^2+V.^2);
                    
                    D = D./ D0;
                    
                    H = 1./((1+D).^n);
                      
                    G = H.*FT_img;
                       
                    output_image = real(ifft2(double(G)));
                    imshow(output_image,[ ], 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=output_image;
                case "Low Pass(Gaussian)" %32
                    input_image = arr{counter};  
                    [M, N] = size(input_image);
                    FT_img = fft2(double(input_image));
                    D0 = app.Input1EditField.Value;  
                    D0 = (D0^2)*2;
                    % Designing filter
                    u = 0:(M-1);
                    idx = find(u>M/2);
                    u(idx) = u(idx)-M;
                    v = 0:(N-1);
                    idy = find(v>N/2);
                    v(idy) = v(idy)-N;
                    
                    [V, U] = meshgrid(v, u);
                    
                    D = sqrt(U.^2+V.^2);
                    
                    D = -D.^2;
                    
                    H = exp(D/D0);
                    
                    G = H.*FT_img;
                      
                    output_image = real(ifft2(double(G)));
                    imshow(output_image,[ ], 'Parent', app.UIAxes_2);
                    counter=counter+1;
                    arr{counter}=output_image;
                otherwise
                    cla(handles.app.UIAxes_2);
                    app.UIAxes_2.Visible = 'off';
                    uialert(app.UIFigure,["No Feature has been selected"; ...
                            "Please select a feature"],'Invalid choice');
            end
            app.Input1EditField.Visible = 'off';
            app.Input1EditFieldLabel.Visible = 'off';
            app.Input2EditField.Visible = 'off';
            app.Input2EditFieldLabel.Visible = 'off';
            function [outImage] = subSampling(image, subSamplingFactor)
                [rows, cols] = size(image);
                outImage = image(1:subSamplingFactor:rows,1:subSamplingFactor:cols);
            end
            function [outImage] = upSampling(image, upSamplingFactor)
                [rows, cols] = size(image);
                newRows = rows*upSamplingFactor;
                newCols = cols*upSamplingFactor;
                rowStart = 1;
                for rowsIndex=1:upSamplingFactor:newRows
                    colStart = 1;
                    for columnIndex=1:upSamplingFactor:newCols
                        outImage(rowsIndex:rowsIndex+upSamplingFactor-1,columnIndex:columnIndex+upSamplingFactor-1) = image(rowStart,colStart);
                        colStart = colStart + 1;
                    end
                    rowStart = rowStart + 1;
                end
            end
        end

        % Button pushed function: ResetButton
        function ResetButtonPushed(app, event)
            global arr;
            global counter;
            if(counter>=2)
                arr{counter}=0;
                counter = counter-1;
                imshow(arr{counter}, 'Parent', app.UIAxes_2);
                if(counter>1)
                    imshow(arr{counter-1}, 'Parent', app.UIAxes);
                end
            end
        end

        % Value changed function: FeaturesDropDown
        function FeaturesDropDownValueChanged(app, event)
            global img;
            value = app.FeaturesDropDown.Value;
            switch value
                case "Sampling Down" %1
                    app.Input1EditField.Visible = 'on';
                    app.Input1EditFieldLabel.Visible = 'on';
                    app.Input2EditField.Visible = 'off';
                    app.Input2EditFieldLabel.Visible = 'off';
                case "Resampling Up" %2
                    app.Input1EditField.Visible = 'on';
                    app.Input1EditFieldLabel.Visible = 'on';
                    app.Input2EditField.Visible = 'off';
                    app.Input2EditFieldLabel.Visible = 'off';
                case "Gray Scale" %3
                    app.Input1EditField.Visible = 'on';
                    app.Input1EditFieldLabel.Visible = 'on';
                    app.Input2EditField.Visible = 'off';
                    app.Input2EditFieldLabel.Visible = 'off';
                case "Log Trans" %7
                    app.Input1EditField.Visible = 'on';
                    app.Input1EditFieldLabel.Visible = 'on';
                    app.Input2EditField.Visible = 'off';
                    app.Input2EditFieldLabel.Visible = 'off';
                case "Inv Log" %8
                    app.Input1EditField.Visible = 'on';
                    app.Input1EditFieldLabel.Visible = 'on';
                    app.Input2EditField.Visible = 'off';
                    app.Input2EditFieldLabel.Visible = 'off';
                case "Power Trans" %9
                    app.Input1EditField.Visible = 'on';
                    app.Input1EditFieldLabel.Visible = 'on';
                    app.Input2EditField.Visible = 'on';
                    app.Input2EditFieldLabel.Visible = 'on';
                case "Contrast Stretch" %10
                    app.Input1EditField.Visible = 'on';
                    app.Input1EditFieldLabel.Visible = 'on';
                    app.Input2EditField.Visible = 'on';
                    app.Input2EditFieldLabel.Visible = 'on';
                case "Threasholding" %11
                    app.Input1EditField.Visible = 'on';
                    app.Input1EditFieldLabel.Visible = 'on';
                    app.Input2EditField.Visible = 'off';
                    app.Input2EditFieldLabel.Visible = 'off';
                case "Gray Scale AP#1" %12
                    app.Input1EditField.Visible = 'on';
                    app.Input1EditFieldLabel.Visible = 'on';
                    app.Input2EditField.Visible = 'on';
                    app.Input2EditFieldLabel.Visible = 'on';
                case "Gray Scale AP#2" %13
                    app.Input1EditField.Visible = 'on';
                    app.Input1EditFieldLabel.Visible = 'on';
                    app.Input2EditField.Visible = 'on';
                    app.Input2EditFieldLabel.Visible = 'on';
                case "Bit-Plane Slicing" %14
                    app.Input1EditField.Visible = 'on';
                    app.Input1EditFieldLabel.Visible = 'on';
                    app.Input2EditField.Visible = 'off';
                    app.Input2EditFieldLabel.Visible = 'off';
                case "Addition" %15
                    app.Input1EditField.Visible = 'on';
                    app.Input1EditFieldLabel.Visible = 'on';
                    app.Input2EditField.Visible = 'off';
                    app.Input2EditFieldLabel.Visible = 'off';
                case "Subtraction"
                    % yes or no question
                    try %if yes
                        app.Input1EditField.Visible = 'on';
                        app.Input1EditFieldLabel.Visible = 'on';
                        app.Input2EditField.Visible = 'off';
                    app.Input2EditFieldLabel.Visible = 'off';
                    catch %if no
                        
                    end
                case "Logic AND"
                    [filename1,filepath1] = uigetfile({'*.*'},'Select Data File 1');
                    cd(filepath1);
                    image=imread([filepath1 filename1]);
                    try
                        img=rgb2gray(image);
                    catch
                    end
                    
                case "High Pass(Ideal)" %27
                    app.Input1EditField.Visible = 'on';
                    app.Input1EditFieldLabel.Visible = 'on';
                    app.Input2EditField.Visible = 'off';
                    app.Input2EditFieldLabel.Visible = 'off';
                case "High Pass(Butterworth)"
                    app.Input1EditField.Visible = 'on';
                    app.Input1EditFieldLabel.Visible = 'on';
                    app.Input2EditField.Visible = 'on';
                    app.Input2EditFieldLabel.Visible = 'on';
                case "High Pass(Gaussian)" %29
                    app.Input1EditField.Visible = 'on';
                    app.Input1EditFieldLabel.Visible = 'on';
                    app.Input2EditField.Visible = 'off';
                    app.Input2EditFieldLabel.Visible = 'off';
                case "Low Pass(Ideal)" %30
                    app.Input1EditField.Visible = 'on';
                    app.Input1EditFieldLabel.Visible = 'on';
                    app.Input2EditField.Visible = 'off';
                    app.Input2EditFieldLabel.Visible = 'off';
                case "Low Pass(Butterworth)"
                    app.Input1EditField.Visible = 'on';
                    app.Input1EditFieldLabel.Visible = 'on';
                    app.Input2EditField.Visible = 'on';
                    app.Input2EditFieldLabel.Visible = 'on';
                case "Low Pass(Gaussian)" %32
                    app.Input1EditField.Visible = 'on';
                    app.Input1EditFieldLabel.Visible = 'on';
                    app.Input2EditField.Visible = 'off';
                    app.Input2EditFieldLabel.Visible = 'off';
                otherwise
                    app.Input1EditField.Visible = 'off';
                    app.Input1EditFieldLabel.Visible = 'off';
                    app.Input2EditField.Visible = 'off';
                    app.Input2EditFieldLabel.Visible = 'off';
            end
        end

        % Button pushed function: RGB2GrayButton
        function RGB2GrayButtonPushed(app, event)
            global image;
            global arr;
            global counter;
            try
                image = rgb2gray(arr{counter});
                imshow(image, 'Parent', app.UIAxes_2);
                counter = counter + 1;
                arr{counter}=image;
            catch
                uialert(app.UIFigure,"The image is already gray",'Invalid choice');
            end
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Color = [0.851 0.502 0.502];
            app.UIFigure.Colormap = [0.2431 0.149 0.6588;0.2431 0.1529 0.6745;0.2471 0.1569 0.6863;0.2471 0.1608 0.698;0.251 0.1647 0.7059;0.251 0.1686 0.7176;0.2549 0.1725 0.7294;0.2549 0.1765 0.7412;0.2588 0.1804 0.749;0.2588 0.1843 0.7608;0.2627 0.1882 0.7725;0.2627 0.1922 0.7843;0.2627 0.1961 0.7922;0.2667 0.2 0.8039;0.2667 0.2039 0.8157;0.2706 0.2078 0.8235;0.2706 0.2157 0.8353;0.2706 0.2196 0.8431;0.2745 0.2235 0.851;0.2745 0.2275 0.8627;0.2745 0.2314 0.8706;0.2745 0.2392 0.8784;0.2784 0.2431 0.8824;0.2784 0.2471 0.8902;0.2784 0.2549 0.898;0.2784 0.2588 0.902;0.2784 0.2667 0.9098;0.2784 0.2706 0.9137;0.2784 0.2745 0.9216;0.2824 0.2824 0.9255;0.2824 0.2863 0.9294;0.2824 0.2941 0.9333;0.2824 0.298 0.9412;0.2824 0.3059 0.9451;0.2824 0.3098 0.949;0.2824 0.3137 0.9529;0.2824 0.3216 0.9569;0.2824 0.3255 0.9608;0.2824 0.3294 0.9647;0.2784 0.3373 0.9686;0.2784 0.3412 0.9686;0.2784 0.349 0.9725;0.2784 0.3529 0.9765;0.2784 0.3569 0.9804;0.2784 0.3647 0.9804;0.2745 0.3686 0.9843;0.2745 0.3765 0.9843;0.2745 0.3804 0.9882;0.2706 0.3843 0.9882;0.2706 0.3922 0.9922;0.2667 0.3961 0.9922;0.2627 0.4039 0.9922;0.2627 0.4078 0.9961;0.2588 0.4157 0.9961;0.2549 0.4196 0.9961;0.251 0.4275 0.9961;0.2471 0.4314 1;0.2431 0.4392 1;0.2353 0.4431 1;0.2314 0.451 1;0.2235 0.4549 1;0.2196 0.4627 0.9961;0.2118 0.4667 0.9961;0.2078 0.4745 0.9922;0.2 0.4784 0.9922;0.1961 0.4863 0.9882;0.1922 0.4902 0.9882;0.1882 0.498 0.9843;0.1843 0.502 0.9804;0.1843 0.5098 0.9804;0.1804 0.5137 0.9765;0.1804 0.5176 0.9725;0.1804 0.5255 0.9725;0.1804 0.5294 0.9686;0.1765 0.5333 0.9647;0.1765 0.5412 0.9608;0.1765 0.5451 0.9569;0.1765 0.549 0.9529;0.1765 0.5569 0.949;0.1725 0.5608 0.9451;0.1725 0.5647 0.9412;0.1686 0.5686 0.9373;0.1647 0.5765 0.9333;0.1608 0.5804 0.9294;0.1569 0.5843 0.9255;0.1529 0.5922 0.9216;0.1529 0.5961 0.9176;0.149 0.6 0.9137;0.149 0.6039 0.9098;0.1451 0.6078 0.9098;0.1451 0.6118 0.9059;0.1412 0.6196 0.902;0.1412 0.6235 0.898;0.1373 0.6275 0.898;0.1373 0.6314 0.8941;0.1333 0.6353 0.8941;0.1294 0.6392 0.8902;0.1255 0.6471 0.8902;0.1216 0.651 0.8863;0.1176 0.6549 0.8824;0.1137 0.6588 0.8824;0.1137 0.6627 0.8784;0.1098 0.6667 0.8745;0.1059 0.6706 0.8706;0.102 0.6745 0.8667;0.098 0.6784 0.8627;0.0902 0.6824 0.8549;0.0863 0.6863 0.851;0.0784 0.6902 0.8471;0.0706 0.6941 0.8392;0.0627 0.698 0.8353;0.0549 0.702 0.8314;0.0431 0.702 0.8235;0.0314 0.7059 0.8196;0.0235 0.7098 0.8118;0.0157 0.7137 0.8078;0.0078 0.7176 0.8;0.0039 0.7176 0.7922;0 0.7216 0.7882;0 0.7255 0.7804;0 0.7294 0.7765;0.0039 0.7294 0.7686;0.0078 0.7333 0.7608;0.0157 0.7333 0.7569;0.0235 0.7373 0.749;0.0353 0.7412 0.7412;0.051 0.7412 0.7373;0.0627 0.7451 0.7294;0.0784 0.7451 0.7216;0.0902 0.749 0.7137;0.102 0.7529 0.7098;0.1137 0.7529 0.702;0.1255 0.7569 0.6941;0.1373 0.7569 0.6863;0.1451 0.7608 0.6824;0.1529 0.7608 0.6745;0.1608 0.7647 0.6667;0.1686 0.7647 0.6588;0.1725 0.7686 0.651;0.1804 0.7686 0.6471;0.1843 0.7725 0.6392;0.1922 0.7725 0.6314;0.1961 0.7765 0.6235;0.2 0.7804 0.6157;0.2078 0.7804 0.6078;0.2118 0.7843 0.6;0.2196 0.7843 0.5882;0.2235 0.7882 0.5804;0.2314 0.7882 0.5725;0.2392 0.7922 0.5647;0.251 0.7922 0.5529;0.2588 0.7922 0.5451;0.2706 0.7961 0.5373;0.2824 0.7961 0.5255;0.2941 0.7961 0.5176;0.3059 0.8 0.5059;0.3176 0.8 0.498;0.3294 0.8 0.4863;0.3412 0.8 0.4784;0.3529 0.8 0.4667;0.3686 0.8039 0.4549;0.3804 0.8039 0.4471;0.3922 0.8039 0.4353;0.4039 0.8039 0.4235;0.4196 0.8039 0.4118;0.4314 0.8039 0.4;0.4471 0.8039 0.3922;0.4627 0.8 0.3804;0.4745 0.8 0.3686;0.4902 0.8 0.3569;0.5059 0.8 0.349;0.5176 0.8 0.3373;0.5333 0.7961 0.3255;0.5451 0.7961 0.3176;0.5608 0.7961 0.3059;0.5765 0.7922 0.2941;0.5882 0.7922 0.2824;0.6039 0.7882 0.2745;0.6157 0.7882 0.2627;0.6314 0.7843 0.251;0.6431 0.7843 0.2431;0.6549 0.7804 0.2314;0.6706 0.7804 0.2235;0.6824 0.7765 0.2157;0.698 0.7765 0.2078;0.7098 0.7725 0.2;0.7216 0.7686 0.1922;0.7333 0.7686 0.1843;0.7451 0.7647 0.1765;0.7608 0.7647 0.1725;0.7725 0.7608 0.1647;0.7843 0.7569 0.1608;0.7961 0.7569 0.1569;0.8078 0.7529 0.1529;0.8157 0.749 0.1529;0.8275 0.749 0.1529;0.8392 0.7451 0.1529;0.851 0.7451 0.1569;0.8588 0.7412 0.1569;0.8706 0.7373 0.1608;0.8824 0.7373 0.1647;0.8902 0.7373 0.1686;0.902 0.7333 0.1765;0.9098 0.7333 0.1804;0.9176 0.7294 0.1882;0.9255 0.7294 0.1961;0.9373 0.7294 0.2078;0.9451 0.7294 0.2157;0.9529 0.7294 0.2235;0.9608 0.7294 0.2314;0.9686 0.7294 0.2392;0.9765 0.7294 0.2431;0.9843 0.7333 0.2431;0.9882 0.7373 0.2431;0.9961 0.7412 0.2392;0.9961 0.7451 0.2353;0.9961 0.7529 0.2314;0.9961 0.7569 0.2275;0.9961 0.7608 0.2235;0.9961 0.7686 0.2196;0.9961 0.7725 0.2157;0.9961 0.7804 0.2078;0.9961 0.7843 0.2039;0.9961 0.7922 0.2;0.9922 0.7961 0.1961;0.9922 0.8039 0.1922;0.9922 0.8078 0.1922;0.9882 0.8157 0.1882;0.9843 0.8235 0.1843;0.9843 0.8275 0.1804;0.9804 0.8353 0.1804;0.9765 0.8392 0.1765;0.9765 0.8471 0.1725;0.9725 0.851 0.1686;0.9686 0.8588 0.1647;0.9686 0.8667 0.1647;0.9647 0.8706 0.1608;0.9647 0.8784 0.1569;0.9608 0.8824 0.1569;0.9608 0.8902 0.1529;0.9608 0.898 0.149;0.9608 0.902 0.149;0.9608 0.9098 0.1451;0.9608 0.9137 0.1412;0.9608 0.9216 0.1373;0.9608 0.9255 0.1333;0.9608 0.9333 0.1294;0.9647 0.9373 0.1255;0.9647 0.9451 0.1216;0.9647 0.949 0.1176;0.9686 0.9569 0.1098;0.9686 0.9608 0.1059;0.9725 0.9686 0.102;0.9725 0.9725 0.0941;0.9765 0.9765 0.0863;0.9765 0.9843 0.0824;1 1 1;1 1 1];
            app.UIFigure.Position = [100 100 868 565];
            app.UIFigure.Name = 'MATLAB App';

            % Create UploadImageButton
            app.UploadImageButton = uibutton(app.UIFigure, 'push');
            app.UploadImageButton.ButtonPushedFcn = createCallbackFcn(app, @UploadImageButtonPushed, true);
            app.UploadImageButton.BackgroundColor = [0 0.6706 0.6706];
            app.UploadImageButton.Position = [750 524 100 22];
            app.UploadImageButton.Text = 'Upload Image';

            % Create ResetButton
            app.ResetButton = uibutton(app.UIFigure, 'push');
            app.ResetButton.ButtonPushedFcn = createCallbackFcn(app, @ResetButtonPushed, true);
            app.ResetButton.BackgroundColor = [0.1412 0.6706 0.6706];
            app.ResetButton.Position = [750 19 100 22];
            app.ResetButton.Text = 'Reset';

            % Create FeaturesDropDownLabel
            app.FeaturesDropDownLabel = uilabel(app.UIFigure);
            app.FeaturesDropDownLabel.HorizontalAlignment = 'right';
            app.FeaturesDropDownLabel.Position = [17 166 53 22];
            app.FeaturesDropDownLabel.Text = 'Features';

            % Create FeaturesDropDown
            app.FeaturesDropDown = uidropdown(app.UIFigure);
            app.FeaturesDropDown.Items = {'Select a feature', 'Sampling Down', 'Resampling Up', 'Gray Scale', 'Identity(linear)', 'Negative(linear)', 'Log Trans', 'Inv Log', 'Power Trans', 'Contrast Stretch', 'Threasholding', 'Gray Scale AP#1', 'Gray Scale AP#2', 'Bit-Plane Slicing', 'Addition', 'Subtraction', 'Logic AND', 'Logic OR', 'Histogram', 'Smoothing(Avg Standerd)', 'Smoothing(Avg Weight)', 'Smoothing(Median)', 'Sharpining(Laplacian)', 'Sharpining(Comp Laplacian)', 'Sharpining(Diag Laplacian)', 'Sharpining(Robert Vertical)', 'Sharpining(Robert Horizontal)', 'Sharpining(Robert ALL)', 'Sharpining(Sobel Vertical)', 'Sharpining(Sobel Horizontal)', 'Sharpining(Sobel ALL)', 'High Pass(Ideal)', 'High Pass(Butterworth)', 'High Pass(Gaussian)', 'Low Pass(Ideal)', 'Low Pass(Butterworth)', 'Low Pass(Gaussian)'};
            app.FeaturesDropDown.ValueChangedFcn = createCallbackFcn(app, @FeaturesDropDownValueChanged, true);
            app.FeaturesDropDown.Position = [85 166 121 22];
            app.FeaturesDropDown.Value = 'Select a feature';

            % Create Input1EditFieldLabel
            app.Input1EditFieldLabel = uilabel(app.UIFigure);
            app.Input1EditFieldLabel.HorizontalAlignment = 'right';
            app.Input1EditFieldLabel.Visible = 'off';
            app.Input1EditFieldLabel.Position = [544 166 42 22];
            app.Input1EditFieldLabel.Text = 'Input 1';

            % Create Input1EditField
            app.Input1EditField = uieditfield(app.UIFigure, 'numeric');
            app.Input1EditField.Visible = 'off';
            app.Input1EditField.Position = [601 166 100 22];

            % Create Input2EditFieldLabel
            app.Input2EditFieldLabel = uilabel(app.UIFigure);
            app.Input2EditFieldLabel.HorizontalAlignment = 'right';
            app.Input2EditFieldLabel.Visible = 'off';
            app.Input2EditFieldLabel.Position = [544 126 42 22];
            app.Input2EditFieldLabel.Text = 'Input 2';

            % Create Input2EditField
            app.Input2EditField = uieditfield(app.UIFigure, 'numeric');
            app.Input2EditField.Visible = 'off';
            app.Input2EditField.Position = [601 126 100 22];

            % Create UIAxes
            app.UIAxes = uiaxes(app.UIFigure);
            title(app.UIAxes, 'Original')
            xlabel(app.UIAxes, '')
            ylabel(app.UIAxes, '')
            app.UIAxes.PlotBoxAspectRatio = [1.67820069204152 1 1];
            app.UIAxes.Visible = 'off';
            app.UIAxes.BackgroundColor = [0.851 0.502 0.502];
            app.UIAxes.Position = [1 204 417 257];

            % Create UIAxes_2
            app.UIAxes_2 = uiaxes(app.UIFigure);
            title(app.UIAxes_2, 'After Enhance')
            xlabel(app.UIAxes_2, '')
            ylabel(app.UIAxes_2, '')
            app.UIAxes_2.PlotBoxAspectRatio = [1.69550173010381 1 1];
            app.UIAxes_2.Visible = 'off';
            app.UIAxes_2.BackgroundColor = [0.851 0.502 0.502];
            app.UIAxes_2.Position = [417 204 421 257];

            % Create ApplyButton
            app.ApplyButton = uibutton(app.UIFigure, 'push');
            app.ApplyButton.ButtonPushedFcn = createCallbackFcn(app, @ApplyButtonPushed, true);
            app.ApplyButton.BackgroundColor = [0.1412 0.6706 0.6706];
            app.ApplyButton.Position = [96 90 100 22];
            app.ApplyButton.Text = 'Apply';

            % Create Image
            app.Image = uiimage(app.UIFigure);
            app.Image.Position = [251 477 335 100];
            app.Image.ImageSource = 'Filters_Web-removebg-preview.png';

            % Create RGB2GrayButton
            app.RGB2GrayButton = uibutton(app.UIFigure, 'push');
            app.RGB2GrayButton.ButtonPushedFcn = createCallbackFcn(app, @RGB2GrayButtonPushed, true);
            app.RGB2GrayButton.BackgroundColor = [0.1412 0.6706 0.6706];
            app.RGB2GrayButton.Position = [96 40 100 22];
            app.RGB2GrayButton.Text = 'RGB2Gray';

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = osama

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end