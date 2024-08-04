clear all
close all
clc

gpuDevice(1);
% return;
Ysr_Model_Epochs=15;
ReTrainFlag=1;


answer = inputdlg('Enter Number of Epochs:',...
             'Epochs', [1 50]);
user_val = str2num(answer{1});
% user_val

Ysr_Model_Epochs=user_val;

Ysr_Model_Epochs
%%
% Transfer Learning Model
% Used_Model='resnet50';
% Ysr_FL='activation_40_relu';

Used_Model='resnet18';
Ysr_FL='res4b_relu';

% Used_Model='resnet101';
% Ysr_FL='res4b22_relu';
%%
% Training Coordinates File

load('Yasser_outdoor_DS_Merged_UrTextV4_2070_OrignialSize_without_augmentation.mat');

stopSigns2=Yasser_outdoor_DS_Merged_Ysr_UrTextV4;
% stopSigns2b = stopSigns2(1:504,[1,5]);    % trains on First & Last column
stopSigns2b = stopSigns2(1:2070,[1,2]);    % trains on First & Last column
% return

%             ---->>>      ------>   Trained on :::   4212-images  <<-----
%%
b=[];
% New_Images_Path='E:\PhD2_Yasser_v2\Detection_Results_For_Papers\Yasser_SSD_Detection_Natural_DataSet\Yasser_Urdu_DataSet_Part1_vol2';
% New_Images_Path='E:\Yasser\Yasser_Chandio2000_Word_Recognition\Yasser_SSD_Detection_Natural_DataSet_v1\New_Volume_2b_images__930X1240';
% New_Images_Path='D:\Yasser_Arafat\Yasser_Natural\Yasser_Urdu_DataSet_Part1_vol2';
New_Images_Path='E:\Yasser\Yasser_Chandio2000_Word_Recognition\Yasser_SSD_Detection_Natural_DataSet_v1\New_Volume_2b_images__930X1240_UrTextV4';

imgRows=930;
imgCols=1240;

inputImageSize=[imgRows imgCols 3];
NewFolderName=['New_Volume_2b_images__UrTextV4_' num2str(imgRows) 'X' num2str(imgCols)];
Yassers_Model=['Yasser_Natural_' NewFolderName '_Urdu_FasterRCNN_FaisalComputer_Trained_On_'];

%%
% Add fullpath to image files.
temp=[];
for kYasser=1:size(stopSigns2b,1)
            [filepath,name,ext] = fileparts((stopSigns2b.Original_Outodoor_Image_Path{kYasser}));  
            %%%%%%%Activate following line if want to copy source images to current Diectory    
            stopSigns2b.Original_Outodoor_Image_Path{kYasser} = [fullfile(New_Images_Path,name) '.jpg'];
            temp=stopSigns2b.Merged_Text_n_MixText_Rects_CoOrdinates{kYasser};
            stopSigns2b.Merged_Text_n_MixText_Rects_CoOrdinates{kYasser}=temp;
end
Yasser_outdoor_DS_Merged_Ysr_UrTextV4=stopSigns2b;
%%
rng('default');
% Randomly split data into a training and test set.
shuffledIndices = randperm(height(Yasser_outdoor_DS_Merged_Ysr_UrTextV4));
idx = floor(0.90 * length(shuffledIndices) );
trainingData = Yasser_outdoor_DS_Merged_Ysr_UrTextV4(shuffledIndices(1:idx),:);
testData = Yasser_outdoor_DS_Merged_Ysr_UrTextV4(shuffledIndices(idx+1:end),:);
% Ysr_Final_Text_CoOrs=(Yasser_Urdu_Text);


imdsTrain=imageDatastore(trainingData.Original_Outodoor_Image_Path);
blds_Train=boxLabelDatastore(trainingData(:,2));
ds_Train=combine(imdsTrain, blds_Train);

imdsTest=imageDatastore(testData.Original_Outodoor_Image_Path);
blds_Test=boxLabelDatastore(testData(:,2));
ds_Test=combine(imdsTest, blds_Test);



%%
preprocessedTrainingData = transform(ds_Train, @(data)preprocessData(data,inputImageSize));
numAnchors = 16;
anchorBoxes = estimateAnchorBoxes(preprocessedTrainingData,numAnchors)

%%
augmentedTrainingData = transform(ds_Train,@augmentData);
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'Rectangle',data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,'BorderSize',10)
disp('Pausing for 3 Seconds.....');
pause(3);

% % % %%
% % % ds_Train = transform(augmentedTrainingData,@(data)preprocessData(ds_Train,inputImageSize));
% % % data = read(ds_Train);
% % % 
% % % figure
% % % I = data{1};
% % % bbox = data{2};
% % % annotatedImage = insertShape(I,'Rectangle',bbox);
% % % annotatedImage = imresize(annotatedImage,2);
% % % figure
% % % imshow(annotatedImage)

%%

%%
% %%   Routine to verify correctness of annotation.
% while hasdata(ds_Test)
%     T = read(ds_Test);
%     RGB = insertObjectAnnotation(T{1},'rectangle',T{2},'YYY--',...
%     'TextBoxOpacity',0.9,'FontSize',18);
%     imshow(RGB)
%     title('Annotated Mine Dataset');
%     pause(0.5);
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


options = trainingOptions('sgdm', ...
    'MiniBatchSize', 1, ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', Ysr_Model_Epochs, ...
    'ExecutionEnvironment','gpu', ...
    'CheckpointPath', [pwd, '\FasterRCNN_Natural_images_Training']);

% % anchorBoxes = [
% %     28,59;16,31;24,21;40,81; 
% %     10,37;77,44;26,41;38,32;
% %     60,94;19,82;16,53;9,19;
% % ];

% % % anchorBoxes = [
% % %     21,    54;
% % %     46,   120;
% % %     22,    36;
% % %     40,    80;
% % %     24,   353;
% % %     16,    75;
% % %     37,    26;
% % %      9,    17;
% % %     13,    44;
% % %     22,    21;
% % %     14,    26;
% % %     25,    85;
% % %     58,    45;
% % %     35,    53;
% % %     10,    28;
% % %     77,   202;
% % %     ];

% [anchorBoxes,meanIoU] = estimateAnchorBoxes(ds_Train,16);

featureExtractionNetwork = Used_Model;
featureLayer = Ysr_FL;
numClasses = width(trainingData)-1;
pause(3);
% return;
lgraph = fasterRCNNLayers(inputImageSize,numClasses,anchorBoxes, ...
                          featureExtractionNetwork,featureLayer)
        if ReTrainFlag==0
                tic
              [detector, info ]  = trainFasterRCNNObjectDetector(trainingData, lgraph, options, ...
                                'NegativeOverlapRange',[0 0.3], ...
                                'PositiveOverlapRange',[0.6 1], ...
                                'TrainingMethod','four-step');
                Y_endTime=toc;
                Y_TrainTime=Y_endTime;
        else
                disp('Retraining from older Check point');
%                 Train_File_Name='Yasser_Natural_New_Volume_2b_images__UrTextV4_930X1240_Urdu_FasterRCNN_FaisalComputer_Trained_On_1863_Tested_On_207-images_n_Model-Name_resnet18_Ep15_Tr_ap-0.5547_Ts_ap-0.43522_5283.9263_.mat'
%                 data=load('faster_rcnn_stage_4_checkpoint__17820__2020_09_13__22_05_35.mat');
     %%           Train_File_Name='faster_rcnn_stage_2_checkpoint__1814__2020_09_19__12_06_37.mat';
% 	         Train_File_Name='faster_rcnn_stage_2_checkpoint__3628__2020_09_19__20_06_57.mat';
        Train_File_Name='faster_rcnn_stage_3_checkpoint__9215__2020_09_20__22_18_15.mat';

                data=load(Train_File_Name);
                tic
                [detector, info ]  = trainFasterRCNNObjectDetector(trainingData, data.detector, options, ...
                              'NegativeOverlapRange',[0 0.3], ...
                              'PositiveOverlapRange',[0.6 1], ...
                              'TrainingMethod','four-step');
                Y_endTime=toc;
                Y_TrainTime=Y_endTime;
        end
         Ysr_Model_Epochs=Ysr_Model_Epochs+15;                          
save(['Yasser_' NewFolderName '_pixels_FasterRCNN_OutDoor_' Used_Model ' _Nabbel_v5.mat'], 'detector','info','Y_TrainTime');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




                %/////////////////////////////////////////////////////////////////////////
                %/////////////////////////////////////////////////////////////////////////////////////
                 %//////////////////////////////////////////////////////////////////
                %//////////////////////////// Training Accuracy//////////////////////////////////////
                %/////////////////////////////////////////////////////////////////////////////////////
                %//////////////////////////////////////////////////////////////////////
                results=[];
                numImages = size(trainingData,1);
                results= struct('Boxes',[],'Scores',[]);
                GroundTruth=table((trainingData.Merged_Text_n_MixText_Rects_CoOrdinates));
                BlackZerosImg=0;
                hold on,
                for i = 1:numImages/1
                                I=imread(trainingData.Original_Outodoor_Image_Path{i});
                                [bboxes,scores] = detect(detector,I,'ExecutionEnvironment','gpu','MiniBatchSize',1,'NumStrongestRegions',30);
                                results(i).Boxes = bboxes;
                                results(i).Scores = scores;
                                disp(['Tr-' num2str(i)]);
                end
                results = struct2table(results);
                [ap_Train,Train_recall,Train_precision] = evaluateDetectionPrecision(results,GroundTruth);
                figure
                plot(Train_recall,Train_precision)
                grid on
                title(sprintf('Train-Set Average Precision = %.4f',ap_Train));
                TrResults={ap_Train,Train_recall,Train_precision};
                TrGroundTruth=GroundTruth;
                %/////////////////////////////////////////////////////////////////////////////////////
                %/////////////////////////////////////////////////////////////////////////////////////


                %//////////////////////////////////////////////////////////////////
                %//////////////////////////// Testing Accuracy//////////////////////////////////////
                %/////////////////////////////////////////////////////////////////////////////////////
                %/////////////////////////////////////////////////////////////////////////////////////
                figure,
                results=[];
                numImages = size(testData,1);
                results= struct('Boxes',[],'Scores',[]);
                GroundTruth=table((testData.Merged_Text_n_MixText_Rects_CoOrdinates));
%                 GroundTruth=table((testData.Merged_Text_n_MixText_Rects_CoOrdinates(1:4)));
                BlackZerosImg=0;
                hold on,
%                 for i = 1:numImages/100
                for i = 1:numImages/1 
                               I =imread(testData.Original_Outodoor_Image_Path{i});
                              
                                [bboxes,scores] = detect(detector,I,'ExecutionEnvironment','gpu','MiniBatchSize',1,'NumStrongestRegions',30);
                                detectedImg = insertShape(I, 'Rectangle', bboxes,'Color','red');
                                imshow(detectedImg)
                                drawnow 
                                pause(0.1);
                                results(i).Boxes = bboxes;
                                results(i).Scores = scores;
                                disp(['Ts-' num2str(i)]);
                                
                                spath=[fullfile(pwd,'FasterRCNN_Test_Detection_Results') '\FasterRCNN_Detections' num2str(i)];
%                                 savefig([fullfile(pwd,'FasterRCNN_Test_Detection_Results') '\FasterRCNN_Detections' num2str(i)])
                                f = gcf;
                                % Requires R2020a or later
%                                 exportgraphics(f,[spath '.png'],'Resolution',150);
                end
                results = struct2table(results);

                % GroundTruth=table((s3.RotatedCoordinates_plus_Angle));
                % [ap,recall,precision] = evaluateDetectionPrecision(results(1:20,:),GroundTruth(1:20,:));
                [ap_Test,Test_recall,Test_precision] = evaluateDetectionPrecision(results,GroundTruth);
                figure
                plot(Test_recall,Test_precision)
                grid on
                title(sprintf('Test-Set Average Precision = %.4f',ap_Test))
                xlabel('Recall');
                ylabel('Precision');
                %/////////////////////////////////////////////////////////////////////////////////////
                %/////////////////////////////////////////////////////////////////////////////////////
                TsResults={ap_Test,Test_recall,Test_precision};
                TsGroundTruth=GroundTruth;
                YsrModel_Name=[Yassers_Model num2str(size(trainingData,1)) '_Tested_On_' num2str(size(testData,1)) '-images_n_Model-Name_' Used_Model '_Ep' num2str(Ysr_Model_Epochs) '_Tr_ap-' num2str(ap_Train) '_Ts_ap-' num2str(ap_Test) '_' num2str(Y_TrainTime) '_.mat'];
                save(YsrModel_Name,'detector','info','TrResults','Ysr_Model_Epochs','TsResults','TrGroundTruth','TsGroundTruth','ap_Train','ap_Test','Y_TrainTime');
                
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

function data = augmentData(data)
% Randomly flip images and bounding boxes horizontally.
tform = randomAffine2d('XReflection',true);
rout = affineOutputView(size(data{1}),tform);
data{1} = imwarp(data{1},tform,'OutputView',rout);
data{2} = bboxwarp(data{2},tform,rout);
end

function data = preprocessData(data,targetSize)
% Resize image and bounding boxes to targetSize.
scale = targetSize(1:2)./size(data{1},[1 2]);
data{1} = imresize(data{1},targetSize(1:2));
data{2} = bboxresize(data{2},scale);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


