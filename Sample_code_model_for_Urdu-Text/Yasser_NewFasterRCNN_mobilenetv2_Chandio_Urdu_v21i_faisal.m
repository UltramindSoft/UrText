clc
close all
clear all
% return
gpuDevice(1);
% data=load('Yasser_MLT2019_Arabic_1K_rectangle_trainings_CurDir_BLI.mat');
% % data=load('ICDAR2019_MLT_images_Arabic_Transformed_1500x1500_Ysr_rectangle_trainings_CurDir3_BLI_included-Ratio_0.80422_.mat');
% trainingData = data.Tab;
% % trainingData = data.Yasser_MLT2019_Arabic;

%load('ICDAR2019_MLT_images_Arabic_Transformed_1400x1400_Ysr_rectangle_trainings_CurDir3_BLI_included-Ratio_0.86709_.mat');
% load('ICDAR2019_MLT_images_Arabic_Transformed_800x800_Ysr_rectangle_trainings_CurDir3_BLI_included-Ratio_0.80215_.mat');
% load('ICDAR2019_MLT_images_Arabic_Transformed_600x600_Ysr_rectangle_trainings_CurDir3_BLI_included-Ratio_0.71197_.mat');
% load(fullfile('E:\Yasser\Yasser_Chandio2000_Word_Recognition\Yasser_Chandio2020_Detection_Prj_v3_','Urdu2020_Chandio__images_Urdu_Transformed_900x900_Ysr_rectangle_trainings_CurDir3_BLI_included-Ratio_0.94924_.mat'));
load('Urdu2020_Chandio__images_Urdu_Transformed_900x900_Ysr_rectangle_trainings_CurDir3_BLI_included-Ratio_0.94924_.mat');
trainingData = Yasser_MLT2019_Arabic;
% imageSize = [2100 2000 3];
% BLI_rows=800;
% BLI_cols=600;

Ysr_scale=2;
NewRowsRatio=Ysr_scale;
NewColsRatio=Ysr_scale;

imgRows=BLI_rows/NewRowsRatio;
imgCols=BLI_cols/NewColsRatio;
imageSize = [imgRows imgCols 3];

trainingData2= trainingData;
% return
% NewPath=fullfile('E:\Yasser\Yasser_Chandio2000_Word_Recognition\Yasser_Chandio2020_Detection_Prj_v3_','Urdu2020_Chandio__images_Urdu_Transformed_450x450');
NewPath='Urdu2020_Chandio__images_Urdu_Transformed_450x450';
for ktk=1:size(trainingData,1)
    [filepath,name,ext] = fileparts(trainingData2.imageFilename{ktk});
%      NewPathTobeAdded= [fullfile(pwd,NewPath) '\' name ext];
     NewPathTobeAdded= [NewPath '\' name ext];
     trainingData2.imageFilename{ktk}=NewPathTobeAdded;
     trainingData2.cor{ktk}= bboxresize(trainingData.cor{ktk}+1,1/Ysr_scale);
%     J = imresize(I,Ysr_scale); 
%     bboxB = bboxresize(bboxA,scale);
end


% rng('default');
% Used_Model='Built_in';
% Used_Model='squeezenet';
% Used_Model='vgg16';    % memory Error
% Used_Model='resnet50';
% Used_Model='alexnet';
% Used_Model='googlenet';
% Used_Model='inceptionv3';
% Used_Model='vgg19';     % Error Nan-Values
% Used_Model='resnet18';
% Used_Model='inceptionresnetv2';  
Used_Model='mobilenetv2';
% Used_Model='YsrModel';
%%%%////////////////////////////////////////////////////////////////////////////////////////////
%%%%////////////////////////////////////////////////////////////////////////////////////////////
%%%%////////////////////////////////////////////////////////////////////////////////////////////
%%%%////////////////////////////////////////////////////////////////////////////////////////////
% % % 
% % % % % % % % Plot the box area versus the box aspect ratio.
% % % % % % % allBoxes = vertcat(trainingData.cor{:});
% % % % % % % 
% % % % % % % % Plot the box area versus box aspect ratio.
% % % % % % % aspectRatio = allBoxes(:,3) ./ allBoxes(:,4);
% % % % % % % area = prod(allBoxes(:,3:4),2);
% % % % % % % 
% % % % % % % figure
% % % % % % % scatter(area,aspectRatio);
% % % % % % % xlabel("Box Area");
% % % % % % % ylabel("Aspect Ratio (width/height)");
% % % % % % % title("Box area vs. Aspect ratio");
% % % % % % % 
% % % % % % % % Select the number of anchor boxes.
% % % % % % % YasserNumAnchors = 15;
% % % % % % % 
% % % % % % % % Cluster using K-Medoids.
% % % % % % % [clusterAssignments, anchorBoxes, sumd] = kmedoids(allBoxes(:,3:4),YasserNumAnchors,'Distance',@iouDistanceMetric);
% % % % % % % 
% % % % % % % % Display estimated anchor boxes. The box format is the [width height].
% % % % % % % anchorBoxes
% % % % % % % % Displayclustering results.
% % % % % % % figure
% % % % % % % gscatter(area,aspectRatio,clusterAssignments);
% % % % % % % title("K-Mediods with "+YasserNumAnchors+" clusters")
% % % % % % % xlabel("Box Area")
% % % % % % % ylabel("Aspect Ratio (width/height)");
% % % % % % % grid
% % % % % % % pause(2);
% % % % % % % 
% % % % % % % maxNumAnchors = 15;
% % % % % % %  for k = 1: maxNumAnchors
% % % % % % %     
% % % % % % %     % Estimate anchors using clustering. 
% % % % % % %     [ClusterAssignments, anchorBoxes2, sumd] = kmedoids (allBoxes (:, 3: 4), k, 'Distance' , @ iouDistanceMetric);
% % % % % % %     
% % % % % % %     % Compute mean IoU.
% % % % % % %     counts = accumarray (clusterAssignments, ones (length (clusterAssignments), 1), [], @ (x) sum (x) -1);
% % % % % % %     meanIoU (k) = mean (1-sumd./(counts));
% % % % % % % end
% % % % % % % 
% % % % % % % figure
% % % % % % % plot (1: maxNumAnchors, meanIoU, '-o' )
% % % % % % % ylabel ( "Mean IoU" )
% % % % % % % xlabel ( "Number of Anchors" )
% % % % % % % title ( "Number of Anchors vs. Mean IoU" )


% Potential_anchorBoxes = estimateAnchorBoxes(preprocessedTrainingData,YasserNumAnchors)

%%%%////////////////////////////////////////////////////////////////////////////////////////////
%%%%////////////////////////////////////////////////////////////////////////////////////////////
%%%%////////////////////////////////////////////////////////////////////////////////////////////
%%%%////////////////////////////////////////////////////////////////////////////////////////////

% Define the number of object classes to detect.
numClasses = width(trainingData2)-1;

% return

anchorBoxes = [

   28,    54;
    97,    73;
    29,    31;
   118,    47;
    66,    52;
    55,    21;
    32,    20;
   153,   100;
    45,    38;
    68,   130;
    78,    31;
    21,    18;
];

YasserEpochs=10;

options = trainingOptions('sgdm',...
          'InitialLearnRate',0.001,...
          'Verbose',true,...
          'MiniBatchSize',1,...
          'MaxEpochs',YasserEpochs,...
          'Shuffle','every-epoch',...
          'VerboseFrequency',50,...
          'GradientThreshold',2,...
          'ExecutionEnvironment','gpu',...
          'CheckpointPath','temp_FasterRCNN_trainings');
%           'CheckpointPath',fullfile('E:\Yasser\Yasser_Chandio2000_Word_Recognition\Yasser_Chandio2020_Detection_Prj_v3_','temp_FasterRCNN_trainings'));

% featureExtractionNetwork = resnet50;
% featureLayer = 'activation_40_relu';

% % featureExtractionNetwork = resnet18;
% % featureLayer = 'res4b_relu';

% featureExtractionNetwork = squeezenet;
% featureLayer = 'fire5-concat';


%%
% Choice of Images to Train .......
NoOfImagesToTrain=1000;
SelectedtrainingData=trainingData2(1:NoOfImagesToTrain,:);
% Randomly split data into a training and test set.
% shuffledIndices = randperm(height(s3));
shuffledIndices = randperm(size(SelectedtrainingData,1));
idx = floor(0.9 * length(shuffledIndices) );
Ysr_trainingData = trainingData2(shuffledIndices(1:idx),:);
Ysr_testData = trainingData2(shuffledIndices(idx+1:end),:);
% return
%%
% % lgraph = fasterRCNNLayers(imageSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);
% tic;
% % % % % FasterRCNN_1400
layers = [ ...
    imageInputLayer(imageSize)
    convolution2dLayer(30,4)  
    leakyReluLayer()
    maxPooling2dLayer(2,'Stride',1)
    
    convolution2dLayer(50,5)  
    leakyReluLayer()
    maxPooling2dLayer(3,'Stride',1)
    
    convolution2dLayer(300,9)  
    reluLayer()
    maxPooling2dLayer(3,'Stride',1)
    
    dropoutLayer(0.2,'Name','drop1')
    
    fullyConnectedLayer(2);
    softmaxLayer()
    classificationLayer()];

% Ysr_trainingData2b =Ysr_trainingData(1:900,:);
Ysr_trainingData2b =Ysr_trainingData;
% Ysr_trainingData2b =Ysr_trainingData(1:70,:);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % return;
% % % disp('.... Returning earlier ...');
% % % return

% numClasses = 1;
% network = 'resnet50';
% featureLayer = 'activation_40_relu';
% anchorBoxes = [64,64; 128,128; 192,192];
% network = 'resnet50';
% featureLayer = 'activation_40_relu';
network = Used_Model;
featureLayer = 'block_13_expand_relu';
lgraph = fasterRCNNLayers(imageSize,numClasses,anchorBoxes, ...
                          network,featureLayer)
% return;

% Retraining the detector ...
% load('Yasser_FasterRCNN_icdar2019_Arabic_v1h_Trained_On_900_Tested_On_100-images_n_Model-Name_YsrModel_Ep5_87386.8293_.mat', 'detector');
% load('Yasser_FasterRCNN1_Chandio2020_Urdu_mobilenetv2_Trained_On_900_Tested_On_100-images_n_Model-Name_YsrModel_Ep35_17224.9013_.mat')
% load('Yasser_FasterRCNN1_Chandio2020_Urdu_mobilenetv2_Trained_On_900_Tested_On_100-images_n_Model-Name_YsrModel_Ep35_17224.9013_.mat')
% load('faster_rcnn_stage_4_checkpoint__2100__2020_09_02__16_56_23.mat')
% load('Yasser_FasterRCNN_Chandio2020_Urdu_mobilenetv2_Trained_On_900_Tested_On_100-images_n_Model-Name_mobilenetv2_Ep5_Tr_ap-0.24839_Ts_ap-0.16658_TrainTime-1406.8149-Seconds_.mat');
% load('Yasser_FasterRCNN_Chandio2020_Urdu_mobilenetv2_Trained_On_900_Tested_On_100-images_n_Model-Name_mobilenetv2_Ep50_Tr_ap-0.23092_Ts_ap-0.22167_TrainTime-3764.1747-Seconds_.mat');
load('Yasser_FasterRCNN_Chandio2020_Urdu_mobilenetv2_Trained_On_900_Tested_On_100-images_n_Model-Name_mobilenetv2_Ep55_Tr_ap-0.23457_Ts_ap-0.22547_TrainTime-5534.4931-Seconds_.mat')

% Ysr_trainingData = transform(Ysr_trainingData,@(data)preprocessData(data,inputImageSize));
tic
% tic
[detector, info] = trainFasterRCNNObjectDetector(Ysr_trainingData, detector , options, ...
    'NegativeOverlapRange', [0 0.1], ...
    'PositiveOverlapRange', [0.66 1], ...
    'SmallestImageDimension', 450, ...
     'TrainingMethod', 'four-step');
%  [detector, info] = trainFasterRCNNObjectDetector(Ysr_trainingData, lgraph , options, ...
%     'NegativeOverlapRange', [0 0.1], ...
%     'PositiveOverlapRange', [0.66 1], ...
%     'SmallestImageDimension', 450, ...
%      'TrainingMethod', 'four-step');
Y_endTime=toc;
Y_TrainTime=Y_endTime;
% YasserEpochs=YasserEpochs+5;
% YsrModel_Name=['e:\Yasser_FasterRCNN_Chandio2020_Urdu_resnet50_Trained_On_' num2str(size(Ysr_trainingData,1)) '_Tested_On_' num2str(size(Ysr_testData,1)) '-images_n_Model-Name_' Used_Model '_Ep' num2str(YasserEpochs) '_' num2str(Y_TrainTime) '_.mat'];
YsrModel_Name=['Yasser_FasterRCNN_Chandio2020_Urdu_mobilenetv2_Trained_On_' num2str(size(Ysr_trainingData,1)) '_Tested_On_' num2str(size(Ysr_testData,1)) '-images_n_Model-Name_' Used_Model '_Ep' num2str(YasserEpochs) '_' num2str(Y_TrainTime) '_.mat'];
save(YsrModel_Name,'detector','info');


%%

%%
% figure,
% plot(info.TrainingLoss)
% grid on
% xlabel('Number of Iterations')
% ylabel('Training Loss for Each Iteration')

figure,
GroundTruthblankImageCounter=0;
NoOfImagesWithAnnotationCounter=0;
NoAnnotationImageCounter=0;
Tr_numImages = size(Ysr_trainingData,1);
    results = table('Size',[Tr_numImages 3],...
        'VariableTypes',{'cell','cell','cell'},...
        'VariableNames',{'Boxes','Scores','Labels'});
    for i = 1:Tr_numImages

        % Read the image.
        I = imread(Ysr_trainingData.imageFilename{i});

%         % Run the detector.
        [bboxes, scores, labels] = detect(detector, I,'ExecutionEnvironment','gpu','MiniBatchSize',1,'NumStrongestRegions',40);
% 
%         %%
%         %%/////////////////////////////////////////////////////////////////
        if ~isempty(Ysr_trainingData.cor{i})
              RGB = insertObjectAnnotation(I,'rectangle',Ysr_trainingData.cor{i},'GT_text',...
             'TextBoxOpacity',0.9,'FontSize',18,'color','y');
              imshow(RGB);
                title('Annotated text rectangles');
              pause(0.1);
        else
            GroundTruthblankImageCounter=GroundTruthblankImageCounter+1
        end
        if ~isempty(bboxes)
            i
             RGB = insertObjectAnnotation(I,'rectangle',bboxes,'PT_text',...
             'TextBoxOpacity',0.9,'FontSize',18,'color','r');
              imshow(RGB);
                title('Detected text rectangles');
              NoOfImagesWithAnnotationCounter=NoOfImagesWithAnnotationCounter+1;
        else
           NoAnnotationImageCounter=NoAnnotationImageCounter+1 ;
        end
        %/////////////////////////////////////////////////////////////////
        
        %%
        % Collect the results.
        % Collect the results.
        results.Boxes{i} = bboxes;
        results.Scores{i} = scores;
        results.Labels{i} = labels;
        disp(['Tr-' num2str(i)]);
        pause(0.05);
    end
Train_GroundTruth=Ysr_trainingData(:,2);
% Extract expected bounding box locations from test data.
% expectedResults = Ysr_trainingData(:,2);
[train_ap_Train,train_recall,train_precision] = evaluateDetectionPrecision(results,Train_GroundTruth);

figure
plot(train_recall,train_precision)
grid on
title(sprintf('Train-Set Average Precision = %.4f',train_ap_Train));
TrResults={train_ap_Train,train_recall,train_precision};
Training_GroundTruth=Train_GroundTruth;
%/////////////////////////////////////////////////////////////////////////////////////
%/////////////////////////////////////////////////////////////////////////////////////


%//////////////////////////////////////////////////////////////////
%//////////////////////////// Testing Accuracy//////////////////////////////////////
%/////////////////////////////////////////////////////////////////////////////////////
%/////////////////////////////////////////////////////////////////////////////////////
% results=[];
Ts_numImages = size(Ysr_testData,1);
results= struct('Boxes',[],'Scores',[]);
    tst_results = table('Size',[Ts_numImages 3],...
        'VariableTypes',{'cell','cell','cell'},...
        'VariableNames',{'Boxes','Scores','Labels'});
hold ,
for i = 1:Ts_numImages

                I = imread(Ysr_testData.imageFilename{i});
                [bboxes,scores,labels] = detect(detector,I,'ExecutionEnvironment','gpu','MiniBatchSize',1,'NumStrongestRegions',40);
                detectedImg = insertShape(I, 'Rectangle', bboxes,'Color','green','LineWidth',5);
                imshow(detectedImg);
                title(['Testing_images: ' num2str(i)]);
                drawnow 
                   tst_results.Boxes{i} = bboxes;
                   tst_results.Scores{i} = scores;
                   tst_results.Labels{i} = labels;
                disp(['Ts-' num2str(i)]);
                pause(0.15);
end
Test_GroundTruth=Ysr_testData(:,2);

%%
% % % % % % % % Removing Empty_Rows
% % % % % % % tst_results2=tst_results;
% % % % % % % ss=Test_GroundTruth;
% % % % % % % idx=all(cellfun(@isempty,ss{:,2}),2);
% % % % % % % ss(idx,:)=[];
% % % % % % % tst_results2(idx,:)=[];   % removing rows from corresponding tesging-results
% % % % % % % Test_GroundTruth2=ss;
%------Example----------------
% % % % v={'a' 1 'k';'b' 2 '';'' 1 'v'}
% % % % w=cell2table(v)
% % % % %------the code---------------
% % % % a=table2cell(w)
% % % % idx=any(ismember(cellfun(@num2str,a,'un',0),''),2)
% % % % w(idx,:)=[]
% [test_ap_Test,test_recall,test_precision] = evaluateDetectionPrecision(tst_results2,Test_GroundTruth2);
[test_ap_Test,test_recall,test_precision] = evaluateDetectionPrecision(tst_results,Test_GroundTruth);
figure
plot(test_recall,test_precision)
grid on
title(sprintf('Test-Set Average Precision = %.4f',test_ap_Test))
xlabel('Recall');
ylabel('Precision');
%/////////////////////////////////////////////////////////////////////////////////////
%/////////////////////////////////////////////////////////////////////////////////////
TsResults={test_ap_Test,test_recall,test_precision};
Testing_GroundTruth=Test_GroundTruth;
YasserEpochs
% YsrModel_Name=['e:\Yasser_FasterRCNN_Chandio2020_Urdu_resnet50_Trained_On_' num2str(size(Ysr_trainingData,1)) '_Tested_On_' num2str(size(Ysr_testData,1)) '-images_n_Model-Name_' Used_Model '_Ep' num2str(YasserEpochs) '_Tr_ap-' num2str(train_ap_Train) '_Ts_ap-' num2str(test_ap_Test) '_TrainTime-' num2str(Y_TrainTime) '-Seconds_.mat'];
YsrModel_Name=['Yasser_FasterRCNN_Chandio2020_Urdu_mobilenetv2_Trained_On_' num2str(size(Ysr_trainingData,1)) '_Tested_On_' num2str(size(Ysr_testData,1)) '-images_n_Model-Name_' Used_Model '_Ep' num2str(YasserEpochs) '_Tr_ap-' num2str(train_ap_Train) '_Ts_ap-' num2str(test_ap_Test) '_TrainTime-' num2str(Y_TrainTime) '-Seconds_.mat'];
 save(YsrModel_Name,'detector','info','TrResults','Training_GroundTruth','YasserEpochs','TsResults','Testing_GroundTruth','train_ap_Train','test_ap_Test','Y_TrainTime');
 
 
 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 function data = preprocessData(data,targetSize)
% Resize image and bounding boxes to the targetSize.
scale = targetSize(1:2)./size(data{1},[1 2]);
data{1} = imresize(data{1},targetSize(1:2));
bboxes = round(data{2});
data{2} = bboxresize(bboxes,scale);
end
