% Function to Calculate Accuracy (for Classification Tasks)
function accuracy = calculateAccuracy(clusterAssignments, trueLabels)
    correct = sum(clusterAssignments == trueLabels);  % Correct predictions
    accuracy = correct / length(trueLabels);
end