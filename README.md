# IIT_GNASSIGNMENT

TASK 1

### **What is Bias-Variance Tradeoff?**
- **Bias** refers to how much your model's predictions differ from the actual values. A **high bias** means your model is too simple, and it makes errors by missing important patterns in the data (this is called **underfitting**).
- **Variance** refers to how much the model's predictions change when trained on different sets of data. A **high variance** means the model is too complex, and it is sensitive to small changes in the data (this is called **overfitting**).

**Bias-Variance Tradeoff** is the balance between having a simple model (which has high bias and underfits) and having a complex model (which has high variance and overfits).



### **What Does This Task Require You to Do?**
You are given **featurized data** (data where features have already been extracted from raw sensor data) and are asked to use it for the following steps:



### **Steps for Task 1:**
1. **Use a Decision Tree Classifier**:
   - A **Decision Tree** is a model that splits data into different parts based on certain criteria, like a flowchart, and makes predictions based on these splits.
   
2. **Vary the Tree's Depth**:
   - The **depth** of a tree refers to how many splits (or decisions) it makes. A **shallow tree** (low depth) will make only a few splits, which might result in **underfitting** because the model will be too simple and won't capture the patterns in the data. On the other hand, a **deep tree** (high depth) will make many splits, which might result in **overfitting** because the model will become too complex and fit the noise in the data, rather than the actual patterns.

3. **Demonstrate the Bias-Variance Tradeoff**:
   - By changing the depth of the tree, you will observe how the model’s performance changes.
     - **Shallow trees (low depth)**: Might not perform well on the training data (underfitting), as they are too simple.
     - **Deep trees (high depth)**: Might perform very well on the training data but perform poorly on new data (overfitting), because they are too complex and memorize the training data.
   
4. **Visualize the Results**:
   - You will create a graph that shows how the model’s performance changes as you change the tree’s depth.
   - You’ll plot the **training accuracy** (how well the model fits the training data) and **testing accuracy** (how well the model generalizes to unseen data).
   
   - **What will you notice?**
     - As the tree gets deeper, the training accuracy will increase because the model becomes more complex and fits the data better.
     - However, the testing accuracy might decrease after a certain point because the model starts to overfit (memorize the data, but not generalize well).
     - The goal is to find the optimal depth where the model performs well on both the training and testing data, showing a good balance between bias and variance.


TASK -2


### **Objective**  
Train and evaluate four classic Machine Learning models using two cross-validation techniques and compare their performance using four key metrics.



### **Models to Train**  
1. **Random Forest Classifier**:  
   - A collection of multiple decision trees that work together to improve accuracy and reduce overfitting.

2. **Decision Tree Classifier**:  
   - A single tree-like structure where data is split based on feature values to make predictions.

3. **Logistic Regression**:  
   - A simple linear model for classification problems that estimates probabilities.

4. **AdaBoost Classifier**:  
   - An ensemble method that improves performance by combining multiple weak models (like shallow decision trees).

---

### **Cross-Validation Techniques**  
1. **K-Fold Cross-Validation (K-Fold CV)**:  
   - The data is split into **K equal parts (folds)**.  
   - The model is trained on **K-1 folds** and tested on the remaining fold.  
   - This process is repeated **K times**, with a different fold used as the test set each time.  
   - The final performance is the **average of the K test results**.  

   - **Why use K-Fold?**  
     Ensures the model is tested on all parts of the data and avoids biased evaluations.

   - **Code Reference**:  
     The `KFold` class from `sklearn` splits the data into 5 parts (`n_splits=5`), ensuring randomness (`shuffle=True`).

2. **Leave-One-Subject-Out Cross-Validation (LOSO-CV)**:  
   - In LOSO, each "subject" (or specific group in the data) is left out as a test set, and the model is trained on the remaining subjects.  
   - This is especially useful when data is grouped by subjects (e.g., in human activity recognition datasets).  
   - It tests how well the model generalizes to unseen subjects.  

   - **Why use LOSO?**  
     It gives a realistic idea of how the model performs on new, unseen individuals/groups.

---

### **Performance Metrics**  
The models will be compared using the following metrics:

1. **Accuracy**:  
   - The proportion of correctly predicted samples out of all predictions.  
   - **Example**: If 80 out of 100 predictions are correct, accuracy = 80%.

2. **Precision**:  
   - The ratio of correctly predicted positive results to all predicted positives.  
   - **Example**: If 10 samples are predicted as positive but only 8 are correct, precision = 8/10.

3. **Recall (Sensitivity)**:  
   - The ratio of correctly predicted positive results to all actual positives.  
   - **Example**: If there are 12 actual positives, but the model correctly predicts 8, recall = 8/12.

4. F1 Score:
This is a balance between precision and recall, giving us a single number to assess both.
If the model is good at finding positives but also correct in most of its predictions, the F1 score will be high.

