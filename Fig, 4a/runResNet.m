clc;
clear;

imageDir = '.\timeFrqImage';
[~,allImages] = Createimds(imageDir,'');
YTest = allImages.Labels;
tag = string(YTest);
taglist = unique(tag);

%% Model Loading and Prediction
resultDir = ".\resNetForPhoneme";
load(fullfile(resultDir,"resNet.mat"));
[YPred,scores] = classify(outputNet,allImages);
acc = sum(YPred == YTest)./numel(YTest);
figure('Name','ConfusionChart','NumberTitle','off');
chart=confusionchart(YTest,YPred);
customOrder=["b","d","p","f","l","m","e","o","u"];
sortClasses(chart,customOrder);

chart.XLabel='PredictedLabels';
chart.YLabel='trueLabels';


%% Accuracy Distribution Chart
accPerData = zeros(length(tag),1);
for i = 1:length(tag)
    accPerData(i) = scores(i,strcmp(taglist,tag{i}));
end
scatterRange = 0.2;

figure('Name','Accuracy Distribution Chart','NumberTitle','off');
hold on
ylim([0 100]);

for i =1:length(taglist)
    ind = find(tag==taglist(i));
    averageAcc(i) = mean(accPerData(ind))*100;
    xAcc = (i-scatterRange):scatterRange*2/(length(ind)-1):(i+scatterRange);
    scatter(xAcc,accPerData(ind)*100);
end

X = 1:length(taglist);
b1 = bar(X,averageAcc,'cyan');
xticks(X)
xticklabels(taglist)
uistack(b1,"bottom");

%% tsne
fullyConnectedLayerName = 'fc';
fullyConnectedActivations = activations(outputNet,allImages,fullyConnectedLayerName);
fullyConnectedActivations = reshape(double(fullyConnectedActivations),length(taglist),[]);
fullyConnectedActivations = fullyConnectedActivations';

Y = tsne(fullyConnectedActivations,"Algorithm","exact","Distance","euclidean");
clr = hsv(length(unique(tag)));
figure('Name','tsne','NumberTitle','off');
gscatter(Y(:,1), Y(:,2),YTest,clr);

%% Training Process Accuracy and Loss
trainData = readmatrix(fullfile(resultDir,'trainingAcc&Loss.csv'));
validationData = readmatrix(fullfile(resultDir,'validationAcc&Loss.csv'));
trainAcc = trainData(:,1);
trainLoss = trainData(:,2);
valAcc = validationData(:,1);
valLoss = validationData(:,2);

figure('Name','Training Process Accuracy and Loss','NumberTitle','off');
subplot(2,3,1);
plot(trainAcc);
title('Training Accuracy');

subplot(2,3,2);
plot(valAcc);
title('Validation Accuracy');

subplot(2,3,3);
plot(trainAcc);
hold on;
plot(valAcc);
legend('Training','Validation');
title('Accuracy');

subplot(2,3,4);
plot(trainLoss);
title('Training Loss');

subplot(2,3,5);
plot(valLoss);
title('Validation Loss');

subplot(2,3,6);
plot(trainLoss);
hold on;
plot(valLoss);
legend('Training','Validation');
title('Loss');

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
