# IIT_GNASSIGNMENT

TASK 1

### Code Explanation in Summary:
1. **`NumPy` and `Pandas`**:
   - Used for working with numerical and tabular data efficiently.
2. **`Matplotlib` and `Seaborn`**:
   - Used for creating visualizations to understand and explore the data.
3. **Scikit-learn Preprocessing**:
   - `StandardScaler` ensures your input features are scaled, making the training more stable and efficient.
4. **Scikit-learn Models**:
   - Decision trees, random forests, AdaBoost, and logistic regression are classifiers used for solving supervised classification tasks.
5. **Cross-Validation**:
   - `KFold` helps split the data multiple times for reliable performance evaluation.
   - `LeaveOneGroupOut` is used when you have groups in your data and want to validate by leaving one group out at a time.
6. **Metrics**:
   - Accuracy, precision, recall, F1-score, and classification reports are tools to evaluate model performance.


### Explanation Summary:
1. **`np.loadtxt`**:
   - Reads data from a text file and loads it into a NumPy array.
   - Each file (e.g., `X_train.txt`, `y_train.txt`) contains specific parts of the dataset.

2. **Dataset Description**:
   - `X_train.txt`: Contains training features (independent variables).
   - `y_train.txt`: Contains the labels (dependent variables) for the training set.
   - `X_test.txt`: Contains test features.
   - `y_test.txt`: Contains the labels for the test set.
   - `subject_train.txt`: Contains the subject IDs (useful for grouped data or per-person analysis).

3. **`astype(int)`**:
   - Converts the loaded data into integers, as the target labels and subject IDs are categorical values.



### Explanation:
1. **`X_train.shape`** and **`y_train.shape`**:
   - These reveal the size of the training dataset.
   - `X_train.shape` gives the total number of samples and features (e.g., `(7352, 561)` means 7352 samples, 561 features).
   - `y_train.shape` confirms the number of labels matches the number of samples.

2. **`X_test.shape`** and **`y_test.shape`**:
   - These show the dimensions of the test dataset.
   - Ensures that test data (`X_test`) and its labels (`y_test`) have consistent sizes.

3. **Purpose**:
   - These print statements help **verify data loading** and ensure no mismatch in feature or label dimensions between training and testing datasets.




### Explanations:
1. **`np.unique(y_train, return_counts=True)`**:
   - Computes the unique activity labels and their frequencies in `y_train`.
   - `unique`: Array of unique activity IDs.
   - `counts`: Corresponding counts of each unique activity.

2. **`sns.barplot`**:
   - Plots a bar graph where the x-axis represents activity IDs and the y-axis shows their counts.

3. **`plt.xticks`**:
   - Maps the numerical activity labels (e.g., 1, 2, 3, ...) to human-readable strings (e.g., "WALKING", "STANDING", etc.).

4. **`plt.grid(axis="y")`**:
   - Adds horizontal grid lines to make it easier to interpret bar heights.

5. **Purpose**:
   - This code helps visualize the distribution of activity labels in the training data, ensuring the dataset is balanced or identifying any class imbalance.



### Key Points:
1. **`StandardScaler()`**:
   - Normalizes the data so that each feature has a mean of `0` and a standard deviation of `1`.
   - Improves model performance, especially for algorithms sensitive to feature scaling (e.g., Logistic Regression, SVM).

2. **`fit_transform(X_train)`**:
   - Computes the mean and standard deviation **from `X_train`**.
   - Scales `X_train` using these values.

3. **`transform(X_test)`**:
   - Applies the same scaling (mean and standard deviation) learned from the training data to `X_test`.
   - Ensures the test set is scaled consistently with the training set.

4. **Why normalize?**:
   - Ensures all features contribute equally to the model training.
   - Prevents features with large numerical ranges from dominating those with smaller ranges.



### Explanation:
1. **`for depth in range(1, 21):`**:
   - Loops through depths from `1` to `20` to control the complexity of the decision tree.
   - Helps identify the optimal `max_depth` where the model balances underfitting and overfitting.

2. **`DecisionTreeClassifier(max_depth=depth, random_state=42)`**:
   - Limits the tree's depth to avoid overfitting on training data.
   - `random_state=42` ensures consistent and reproducible results.

3. **`clf.fit(X_train_normalized, y_train)`**:
   - Trains the model on the normalized training data.

4. **`clf.score()`**:
   - Calculates the accuracy of the model on the provided data:
     - `clf.score(X_train_normalized, y_train)`: Training accuracy.
     - `clf.score(X_test_normalized, y_test)`: Test accuracy.

5. **Appending to Lists**:
   - `train_accuracy.append(...)`: Collects training accuracy scores for each depth.
   - `test_accuracy.append(...)`: Collects test accuracy scores for each depth.

---

### Purpose:
- This code evaluates the performance of a **Decision Tree Classifier** for different `max_depth` values.
- By comparing training and test accuracy, you can analyze:
   - **Underfitting** (low accuracy on both training and test data at low depths).
   - **Overfitting** (high training accuracy but low test accuracy at larger depths).
   - Identify the optimal depth where the model generalizes well.




### Key Points:
1. **`plt.plot()`**:
   - Plots **training** and **testing** accuracies against tree depth.
   - Helps visualize the model's performance as complexity (tree depth) increases.

2. **Bias-Variance Tradeoff**:
   - **Training Accuracy**: Measures how well the model fits the training data.
   - **Testing Accuracy**: Measures how well the model generalizes to unseen data.
   - The tradeoff shows:
     - **Underfitting** at shallow depths: Low accuracy on both training and testing data.
     - **Overfitting** at large depths: High training accuracy but low test accuracy.

3. **Visualization**:
   - The **grid** and **markers** improve the clarity of the plot.
   - This plot helps identify the optimal tree depth where testing accuracy is maximized while avoiding overfitting. 




TASK -2

