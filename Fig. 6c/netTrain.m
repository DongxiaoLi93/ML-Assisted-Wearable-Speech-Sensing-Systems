clear;
clc;
close all;

%% Data Input
includePattern = '*x';
mainFolder = '.\timeFrqImage';
[testImages,trainImages] = Createimds(mainFolder,includePattern);

%% Set Partitioning
numTrainPercentage = 0.7;
[imdsTrain,imdsValidation] = splitEachLabel(trainImages,numTrainPercentage,"randomize");

%% Network Architecture
nTags=10;
layers = myResNet50(nTags);

%% Configuring Training Options
trainingData = {};
options = trainingOptions("adam", ...
    'OutputFcn',@recordTrainingProgress,...
    InitialLearnRate=0.001, ... 
    MaxEpochs=60, ...       
    MiniBatchSize=256, ...   
    ExecutionEnvironment="gpu", ...
    Shuffle="every-epoch", ...
    ValidationData=imdsValidation, ...
    Plots="training-progress", ...
    ValidationFrequency=1, ...
    Verbose=false);
%     Plots="training-progress", ...
    
global allResults;
allResults = {};

%% Train
numRe = 10;
allNet = {};
allValAccuracy = [];
allTestAccuracy = [];
testAccuracy = [];
for i = 1:numRe
    net = trainNetwork(imdsTrain,layers,options);

    valYPred = classify(net, imdsValidation);
    valYTrue = imdsValidation.Labels;
    valAccuracy = sum(valYPred == valYTrue)/numel(valYTrue);

    if ~isempty(testImages.Files)
        testYPred = classify(net, testImages);
        testYTrue = testImages.Labels;
        testAccuracy = sum(testYPred == testYTrue)/numel(testYTrue);
    end

    allNet = [allNet;net];
    allValAccuracy = [allValAccuracy;valAccuracy];
    allTestAccuracy = [allTestAccuracy;testAccuracy];

end

%% Result
[maxValAccuracy,maxValIndex] = max(allValAccuracy);
averageValAccuracy = sum(allValAccuracy)/length(allValAccuracy);

disp(['Average validation accuracy：',num2str(averageValAccuracy)]);
disp(['Maximum validation accuracy：',num2str(maxValAccuracy),',index:',num2str(maxValIndex)]);


if ~isempty(testImages.Files)
    [maxTestAccuracy,maxTestIndex] = max(allTestAccuracy);
    averageTestAccuracy = sum(allTestAccuracy)/length(allTestAccuracy);
    
    disp(['Average test accuracy：',num2str(averageTestAccuracy)]);
    disp(['Maximum test accuracy：',num2str(maxTestAccuracy),',index:',num2str(maxTestIndex)]);
end

result = {};
for i = 1:length(allResults)
    resultTemp = allResults(i);
    resultTemp = resultTemp{1,1};
    result = [result;{allNet(i),resultTemp.trainAcc,resultTemp.trainLoss,resultTemp.valAcc,resultTemp.valLoss}];
end
%% Result Output
outputFlag = false;
outputChoose = 2;
if outputFlag == true
    reslutSaveMainFolder = '.\';
    folderName = strcat('resNetForSentence', datestr(datetime('now'),'yyyy-mm-dd_HH-MM-SS'));
    mkdir(fullfile(reslutSaveMainFolder,folderName));

    outputData = [result{outputChoose,2},result{outputChoose,3}];
    outputFileName = fullfile(reslutSaveMainFolder,folderName,'trainingAcc&Loss.csv');
    writematrix(outputData,outputFileName);

    outputData = [result{outputChoose,4},result{outputChoose,5}];
    outputFileName = fullfile(reslutSaveMainFolder,folderName,'validationAcc&Loss.csv');
    writematrix(outputData,outputFileName);
    
    outputNet = allNet(i);
    savePath = fullfile(reslutSaveMainFolder,folderName,'resNet');
    save(savePath,'outputNet');
end


%% helper function
function stop = recordTrainingProgress(info)
    persistent epochData;
    persistent trainLoss;
    persistent trainAcc;
    persistent valLoss;
    persistent valAcc;
    global allResults;

    stop = false;  

    if info.State == "start"
        epochData = [];
        trainLoss = [];
        trainAcc = [];
        valLoss = [];
        valAcc = [];
    elseif info.State == "iteration"
        epochData = [epochData; info.Epoch];
        trainLoss = [trainLoss; info.TrainingLoss];
        trainAcc = [trainAcc; info.TrainingAccuracy];
        valLoss = [valLoss; info.ValidationLoss];
        valAcc = [valAcc; info.ValidationAccuracy];
    elseif info.State == "done"
        if isempty(allResults)
            allResults = {};
        end
        resultStruct = struct('Epoch', epochData, 'trainLoss', trainLoss, 'trainAcc', trainAcc,'valLoss',valLoss,'valAcc',valAcc);
        allResults{end+1} = resultStruct;
    end
end

function [includeImds,otherImds] = Createimds(mainFolderPath,includePattern)

includeImds = imageDatastore({});
otherImds = imageDatastore({});

subfolders = dir(mainFolderPath);
subfolders = subfolders(3:end);

for i = 1:length(subfolders)
    subfolderPath = fullfile(mainFolderPath, subfolders(i).name);
    images = imageDatastore(subfolderPath);
    [~,fileName,fileExt] = fileparts(subfolderPath); 
    tempTag = fileName;
    if strfind(tempTag,'_00')
        tempTag=tempTag(1:strfind(tempTag,'_')-1);
    end

    if ~isempty(regexp(subfolders(i).name, includePattern, 'once'))
        includeImds = imageDatastore(cat(1, includeImds.Files, images.Files),...
            'Labels', cat(1, includeImds.Labels, repmat({tempTag}, length(images.Files), 1))); 
    else

        otherImds = imageDatastore(cat(1, otherImds.Files, images.Files),...
            'Labels', cat(1, otherImds.Labels, repmat({tempTag}, length(images.Files), 1))); 
    end
end

includeImds.Labels = categorical(includeImds.Labels);
otherImds.Labels = categorical(otherImds.Labels);

end