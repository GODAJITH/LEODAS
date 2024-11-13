employee_names <- c("John Doe", "Jane Smith", "Peter Johnson", "Emily Davis", 
"Michael Brown") 
 
employee_ages <- c(30, 25, 45, 40, 35) 
 
employee_salaries <- c(55000, 62000, 75000, 68000, 72000) 
 
# Step 2: Combine vectors into a data frame 
 
employee_data <- data.frame(Name = employee_names, Age = employee_ages, 
Salary = employee_salaries) 
 
# Step 3: Display the data frame 
 
print("Employee Data:") 
 
print(employee_data) 
************************************************************************************************** exp 2 
# Create a list of students with their corresponding grades 
students_list <- list("Alice" = "A", "Bob" = "B", "Charlie" = "C") 
 
# Display the original list 
cat("Original List of Students and Grades:\n") 
print(students_list) 
 
# Adding a new student to the list 
students_list$David <- "B+" 
cat("\nAfter adding a new student (David) with grade B+:\n") 
print(students_list) 
 
# Removing a student from the list 
students_list$Bob <- NULL 
cat("\nAfter removing the student Bob:\n") 
print(students_list) 
 
# Update an existing student's grade 
students_list$Alice <- "A+" 
cat("\nAfter updating Alice's grade to A+:\n") 
print(students_list) 
 
# Checking if a student is in the list 
if ("Charlie" %in% names(students_list)) { 
  cat("\nCharlie is in the list. His grade is:", students_list$Charlie, "\n") 
} else { 
  cat("\nCharlie is not in the list.\n") 
} 
 
# Display the final list of students and grades 
cat("\nFinal List of Students and Grades:\n") 
print(students_list) 
 
Creating and Manipulating an Array 
# Create a 3x3 array of student scores 
students_scores <- array(c(85, 90, 75, 88, 92, 80, 78, 89, 91), dim = c(3, 3),  
                         dimnames = list(c("Alice", "Bob", "Charlie"),  
                                         c("Math", "Science", "English"))) 
 
# Display the original array 
cat("\nOriginal Array of Student Scores:\n") 
print(students_scores) 
 
# Modify a score in the array (update Bob's Science score) 
students_scores["Bob", "Science"] <- 95 
cat("\nAfter updating Bob's Science score to 95:\n") 
print(students_scores) 
 
# Add a new student's scores to the array (David) 
new_student_scores <- c(82, 88, 84) # Scores for Math, Science, and English 
students_scores <- rbind(students_scores, David = new_student_scores) 
cat("\nAfter adding a new student (David) with scores:\n") 
print(students_scores) 
 
# Removing a student from the array (Charlie) 
students_scores <- students_scores[-which(rownames(students_scores) == "Charlie"),] 
cat("\nAfter removing Charlie from the array:\n") 
print(students_scores) 
 
# Display the final array of student scores 
cat("\nFinal Array of Student Scores:\n") 
print(students_scores)
************************************************************************************************************ exp 3
# Create a Data Frame 
students_df <- data.frame( 
  Name = c("Alice", "Bob", "Charlie", "David"), 
  Math = c(85, 88, 78, 92), 
  Science = c(90, 95, 89, 88), 
  English = c(75, 80, 91, 84), 
  Total = c(85+90+75, 88+95+80, 78+89+91, 92+88+84) 
) 
 
# Display the original Data Frame 
cat("Original Data Frame:\n") 
print(students_df) 
 
# Access specific columns (like a matrix) 
cat("\nAccessing the 'Math' column:\n") 
print(students_df$Math) 
 
# Access specific rows (like a matrix) 
cat("\nAccessing the second row (Bob's marks):\n") 
print(students_df[2, ]) 
 
# Sum of marks in each subject (column-wise sum) 
subject_sums <- colSums(students_df[, 2:4]) 
cat("\nSum of marks in each subject:\n") 
print(subject_sums) 
 
# Calculate the average marks for each student 
students_df$Average <- rowMeans(students_df[, 2:4]) 
cat("\nAfter calculating average marks for each student:\n") 
print(students_df) 
 
# Add a grade column based on Total marks 
students_df$Grade <- ifelse(students_df$Total >= 250, "A",  
                            ifelse(students_df$Total >= 230, "B", "C")) 
cat("\nAfter adding a grade column based on total marks:\n") 
print(students_df) 
 
# Modify a specific value (e.g., update Charlie's Math score) 
students_df[students_df$Name == "Charlie", "Math"] <- 80 
students_df$Total <- rowSums(students_df[, 2:4])  # Recalculate Total after modification 
cat("\nAfter updating Charlie's Math score and recalculating total:\n") 
print(students_df) 
 
# Final Data Frame after all operations 
cat("\nFinal Data Frame:\n") 
print(students_df)
************************************************************************************ exp 4
# Load necessary library 
library(dplyr) 
 
# Create a dataset containing customer names and email addresses 
customer_data <- data.frame( 
  Name = c("Alice Johnson", "Bob Smith", "Charlie Brown"), 
  Email = c("alice.johnson@example.com", "bob.smith@example.net", 
"charlie.brown@example.org") 
) 
 
# Convert names to uppercase 
customer_data$Name_Upper <- toupper(customer_data$Name) 
 
# Extract domain names from email addresses 
customer_data$Domain <- sub(".*@", "", customer_data$Email) 
 
# Display the modified dataset 
print(customer_data) 
******************************************************************************************* exp 5-a
# Load necessary library 
library(dplyr) 
 
# Create a dataset containing customer names and email addresses 
customer_data <- data.frame( 
  Name = c("Alice Johnson", "Bob Smith", "Charlie Brown"), 
  Email = c("alice.johnson@example.com", "bob.smith@example.net", 
"charlie.brown@example.com") 
) 
 
# Split full names into first and last names 
customer_data <- customer_data %>% 
  mutate( 
    First_Name = sapply(strsplit(as.character(Name), " "), `[`, 1), 
    Last_Name = sapply(strsplit(as.character(Name), " "), `[`, 2) 
  ) 
 
# Extract domain names from email addresses 
customer_data$Domain <- sub(".*@", "", customer_data$Email) 
 
# Check if the email domain matches a specific domain (e.g., "example.com") 
target_domain <- "example.com" 
customer_data$Is_Target_Domain <- customer_data$Domain == target_domain 
 
# Display the modified dataset 
print(customer_data)  
********************************************************************************************** 5-b
ram: 
# Step 1: Create a sample dataset 
data <- data.frame( 
  Name = c("Alice", "Bob", "Charlie"), 
  Age = c(25, 30, 35), 
  Height = c(5.5, 6.0, 5.8) 
) 
# Step 2: Display the original dataset 
print("Original Dataset:") 
print(data) 
# Step 3: Transpose the dataset using the t() function 
transposed_data <- t(data) 
# Step 4: Convert the transposed matrix back to a dataframe (optional) 
transposed_df <- as.data.frame(transposed_data) 
# Step 5: Display the transposed data 
print("Transposed Dataset:") 
print(transposed_df)
***************************************************************************************************  6
# Step 1: Define parameters for the Normal Distribution (mean = 0, standard deviation = 
1) 
mean_value <- 0 
sd_value <- 1 
# Step 2: Generate 1000 random samples from the normal distribution 
random_data <- rnorm(n = 1000, mean = mean_value, sd = sd_value) 
# Step 3: Display summary statistics of the generated data 
print("Summary Statistics:") 
summary(random_data) 
# Step 4: Plot the data to visualize the distribution 
hist(random_data, breaks = 30, probability = TRUE,  
     main = "Normal Distribution (mean = 0, sd = 1)", 
     xlab = "Value", col = "lightblue") 
 
# Overlay the density curve 
curve(dnorm(x, mean = mean_value, sd = sd_value),  
      col = "darkblue", lwd = 2, add = TRUE)
      *************************************************************************************************  7
      # Sample dataset of heights in centimeters 
heights <- c(150, 160, 165, 170, 175, 180, 185, 190, 195, 200) 
# 1. Mean of heights 
mean_height <- mean(heights) 
cat("Mean Height:", mean_height, "\n") 
# 2. Median of heights 
median_height <- median(heights) 
cat("Median Height:", median_height, "\n") 
# 3. Standard deviation of heights 
sd_height <- sd(heights) 
cat("Standard Deviation of Height:", sd_height, "\n") 
# Example for correlation if there's another variable, e.g., weight 
# Sample dataset of weights corresponding to the heights 
weights <- c(55, 60, 63, 67, 70, 73, 76, 80, 85, 90) 
 
# 4. Correlation between height and weight 
correlation_hw <- cor(heights, weights) 
cat("Correlation between Height and Weight:", correlation_hw, "\n") 
**************************************************************************************************** exp 8
library(ggplot2) 
library(dplyr) 
months <- rep(month.name, each = 3) 
categories <- rep(c("Electronics", "Clothing", "Groceries"), times = 12) 
regions <- c("North", "South", "East", "West") 
set.seed(42) 
sales_data <- data.frame( 
  Month = factor(months, levels = month.name), 
  Category = categories, 
  Sales = round(runif(36, min = 1000, max = 5000)) 
) 
top_category <- sales_data %>% 
  group_by(Category) %>% 
  summarise(TotalSales = sum(Sales)) %>% 
  arrange(desc(TotalSales)) %>% 
  slice(1) %>% 
  pull(Category) 
regional_sales <- data.frame( 
  Region = rep(regions, each = 12), 
  Month = factor(rep(month.name, times = 4), levels = month.name), 
  Sales = round(runif(48, min = 500, max = 4000)), 
  Category = top_category 
) 
line_chart <- ggplot(sales_data, aes(x = Month, y = Sales, color = Category, group = Category)) + 
  geom_line(size = 1) + 
  geom_point() + 
  labs(title = "Monthly Sales Trends for Each Product Category", 
       x = "Month", y = "Sales") + 
  theme_minimal() 
print(line_chart) 
stacked_bar_chart <- ggplot(regional_sales, aes(x = Region, y = Sales, fill = Month)) + 
  geom_bar(stat = "identity") + 
  labs(title = paste("Sales Distribution Among Regions for", top_category, "Category"), 
       x = "Region", y = "Sales") + 
  theme_minimal() + 
  scale_fill_brewer(palette = "Set3") 
print(stacked_bar_chart)
************************************************************************************************* exp 9
library(caret) 
library(randomForest) 
library(ROCR) 
set.seed(42) 
customer_data <- data.frame( 
  age = round(runif(100, 20, 60)), 
  income = round(runif(100, 30000, 100000)), 
  spending_score = round(runif(100, 1, 100)), 
  is_high_risk = sample(c(0, 1), 100, replace = TRUE) # 0 = Low Risk, 1 = High Risk 
) 
 
customer_data$is_high_risk <- as.factor(customer_data$is_high_risk) # Convert target to 
factor for classification 
set.seed(42) 
trainIndex <- createDataPartition(customer_data$is_high_risk, p = 0.8, list = FALSE) 
train_data <- customer_data[trainIndex, ] 
test_data <- customer_data[-trainIndex, ] 
rf_model <- randomForest(is_high_risk ~ ., data = train_data, ntree = 100, mtry = 2, 
importance = TRUE) 
test_pred <- predict(rf_model, test_data) 
conf_matrix <- confusionMatrix(test_pred, test_data$is_high_risk) 
accuracy <- conf_matrix$overall['Accuracy'] 
precision <- conf_matrix$byClass['Pos Pred Value'] 
recall <- conf_matrix$byClass['Sensitivity'] 
pred_prob <- predict(rf_model, test_data, type = "prob")[, 2] 
pred <- prediction(pred_prob, as.numeric(test_data$is_high_risk) - 1) 
perf <- performance(pred, "tpr", "fpr") 
plot(perf, col = "blue", main = "ROC Curve for High-Risk Customer Prediction") 
auc <- performance(pred, "auc")@y.values[[1]] 
cat("AUC:", auc, "\n") 
cat("Model Accuracy:", accuracy, "\n") 
cat("Model Precision:", precision, "\n") 
cat("Model Recall:", recall, "\n") 
cat("AUC-ROC Score:", auc, "\n") 
sample_predictions <- data.frame( 
  Actual = test_data$is_high_risk[1:5], 
  Predicted = test_pred[1:5] 
) 
print(sample_predictions) 
*********************************************************************************************** exp 10