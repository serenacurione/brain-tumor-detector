function metrics = evaluate_metrics(y_true, y_pred)
%   Output:
%       metrics - Struct with fields:
%                 .confMat     - Confusion matrix (C x C)
%                 .accuracy    - Overall accuracy
%                 .sensitivity - Mean per-class sensitivity (recall)
%                 .specificity - Mean per-class specificity
%                 .precision   - Mean per-class precision
%                 .f1          - Mean per-class F1 score
%                 .classNames  - Cell array of class name strings

    classes = unique([y_true; y_pred]);
    classNames = cellstr(string(classes));
    C = length(classes);
    confMat = confusionmat(y_true, y_pred);

    sensitivity = zeros(C, 1);
    specificity = zeros(C, 1);
    precision = zeros(C, 1);
    f1 = zeros(C, 1);

    for i = 1:C
        TP = confMat(i, i);
        FP = sum(confMat(:, i)) - TP;
        FN = sum(confMat(i, :)) - TP;
        TN = sum(confMat(:)) - TP - FP - FN;

        sensitivity(i) = TP / (TP + FN + eps);
        specificity(i) = TN / (TN + FP + eps);
        precision(i) = TP / (TP + FP + eps);
        f1(i) = 2 * TP / (2 * TP + FP + FN + eps);
    end

    metrics.confMat = confMat;
    metrics.accuracy = trace(confMat) / sum(confMat(:));
    metrics.sensitivity = mean(sensitivity);
    metrics.specificity = mean(specificity);
    metrics.precision = mean(precision);
    metrics.f1 = mean(f1);
    metrics.classNames = classNames;
    metrics.perClass = struct('sensitivity', sensitivity, 'specificity', specificity, 'precision', precision, 'f1', f1);
end
