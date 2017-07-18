    detector = vision.CascadeObjectDetector('morenegatives.xml');  
    % xml file of trained data with 23k negative images.
    detector_digit = vision.CascadeObjectDetector('digit.xml');   
    % xml file of images having only digit of speed sign board
    detector2 = vision.CascadeObjectDetector('smallbbox.xml');  
    % xml file of images having small bounding boxes for signals of small size in image 
    srcFiles = dir('F:\New folder\test\*.jpg');      
    % file where test image are stored
    workingDir = 'F:\New folder\test';              
    % directory of test file 
    mkdir(workingDir,'final');                  
    % directoy where results of test images will be stored 
    q = struct();
    imgs = imread('testimage.png');                 
    % test image is used if multiple bounding boxes are found in a single image
    imgs = imresize(imgs,[50,50]);                    
    imgshog = extractHOGFeatures(imgs);
    % running script through all the test images in a single go. 
        for i = 1:length(srcFiles)
        filename_test = strcat('F:\New Folder\test\',srcFiles(i).name);
        img = imread(filename_test);
        p = zeros(200,4);
        r = zeros(200,4);
        t = 1; e = 1;
     
        % the main idea behind using three different detector was, that HOG
        % features are geometry descriptors so any object having similar 
        % as that of speed signs (i.e. circular) will be detected in image.
        % So, first detector detects all the possible things similar to
        % speed signs, be it other traffic regulation sign. Then second
        % detector come into play i.e detector_digit which detects if
        % digits similar to the sign board are present in the bounding box
        % or not. 
        
        
         a = step(detector,img);              
         % speed signs are searched in image using features of detector.
               for k = 1:size(a,1)
                 if a(k,3)<=50 || a(k,4)<=50          
                    % if the bounding box in smaller than [50,50] detector2 
                    % is used to check the content of bounding box.
                    b = img(a(k,2):a(k,2)+a(k,4)+2,a(k,1):a(k,1)+a(k,3)+2);
                    c = step(detector2,b);            
                 else
                    % if bounding boxes are bigger than [50,50]
                    % detector_digit is used to check the content of
                    % bounding box 
                    b = img(a(k,2):a(k,2)+a(k,4)+3,a(k,1):a(k,1)+a(k,3)+3);
                    c = step(detector_digit,b);
                 end
                       for l = 1:size(c,1)                     
                          p(t,:) = c(l,:);            % position of bounding box found in c are stored in p. 
                          t = t+1;
                          r(e,:) = a(k,:);            % position of bounding boxes found in a are stored in r. 
                          e = e+1;
                       end
               end
               
        % if only one box is found using first detector then results of
        % second and third detector are not considered, and the only
        % bounding box is diplayed. This might increase the number of
        % wrongly detected boxes but to check the performance of first
        % detector this method was used.
        
        if size(a,1) == 1                            
        detectedImg2= insertObjectAnnotation(img,'rectangle',a,'speed sign');
        % detected sign image is stored in new directory.
        filename = [sprintf('%03d',i) '.jpg'];                      
        fullname = fullfile(workingDir,'final',filename);
        imwrite(detectedImg2,fullname)
        q(i).imageFilename = filename_test;
        q(i).bbox = a;
        else 
        % if more than one boxes are detected using firt detector results
        % of first, second and third detector are combinly used and is made
        % sure that, correct box is diplayed, out of all the boxes detected by first
        % detector.  
        q(i).imageFilename = filename_test;
        [x,y] = max(r(:,3));
        q(i).bbox = r(y,:);  
        detectedImg2= insertObjectAnnotation(img,'rectangle',r(y,:),'speed sign'); 
        filename = [sprintf('%03d',i) '.jpg'];
        fullname = fullfile(workingDir,'final',filename);
        imwrite(detectedImg2,fullname)
        end
        % if second and third detector results show that none of the boxes
        % detected by first detector contains the subject, a last and final
        % step of test image is carried out, where we test the features of
        % bounding boxes with a test image (imgs as stated above), this was
        % all done because of the lack of positive images provided by the
        % company during the competition. 
        if r == zeros()
            if size(a,1)~=0  
                u =zeros();
                    for m = 1:size(a,1)
                         b = img(a(m,2):a(m,2)+a(m,4)+2,a(m,1):a(m,1)+a(m,3)+2);
                         imgs1 = imresize(b,[50,50]);
                         w = extractHOGFeatures(imgs1);
                         u(m) = pdist2(w,imgshog);
                    end

                    [im,ind] = min(u);
                    n = [a(ind,1) a(ind,2) a(ind,3) a(ind,4)];
                    detectedImg2= insertObjectAnnotation(img,'rectangle',n,'speed sign');
                    filename = [sprintf('%03d',i) '.jpg'];
                    fullname = fullfile(workingDir,'final',filename);
                    imwrite(detectedImg2,fullname)
                    q(i).imageFilename = filename_test;
                    q(i).bbox = n;
            end 
        end
        end
        filesname = 'coordinates.csv';  % file where coordinates of bounding box is stored. 
        q = struct2table(q);
        writetable(q,filesname);


