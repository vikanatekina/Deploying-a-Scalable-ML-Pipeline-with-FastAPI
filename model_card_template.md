# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest Classifier trained on the U.S. Census Income dataset. The model predicts whether an individual's annual income exceeds $50,000 based on demographic and employment-related features. The model was developed as part of the Udacity Deploying a Scalable Machine Learning Pipeline with FastAPI project.

## Intended Use

The model is intended for educational purposes to demonstrate machine learning model development, evaluation, deployment, and monitoring. The model predicts income categories (>50K or <=50K) using census demographic data. It should not be used for real hiring, lending, insurance, or other high-impact decision-making processes.

## Training Data

The training data consists of records from the U.S. Census Income dataset (Adult Census dataset). Features include age, work class, education, marital status, occupation, relationship, race, sex, hours worked per week, and other demographic attributes. The dataset was split into training and testing subsets before model training.

## Evaluation Data

The evaluation dataset was created using a train-test split from the original census dataset. The test dataset was not used during training and was reserved for model evaluation.

## Metrics

The model was evaluated using Precision, Recall, and F1 Score.

Performance on the test dataset:

* Precision: 0.7353
* Recall: 0.6378
* F1 Score: 0.6831

In addition, model performance was evaluated across categorical feature slices and stored in the slice_output.txt file to identify differences in performance among demographic groups.

## Ethical Considerations

This dataset contains demographic attributes such as race, sex, marital status, and nationality. Models trained on this data may inherit historical biases present in the dataset. Predictions may differ across demographic groups and should be carefully evaluated before any practical use. The model should only be used for educational and demonstration purposes.

## Caveats and Recommendations

The model was trained on historical census data and may not generalize well to current populations or different geographic regions. Performance varies across demographic slices, and further fairness analysis is recommended before any operational use. Future improvements could include hyperparameter tuning, additional feature engineering, bias analysis, and more comprehensive validation techniques.
