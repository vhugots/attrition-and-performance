#################################################
# Create attrition train set and validation set #
#################################################

## Load required libraries and installing if they are not
if (!require(tidyverse))
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(caret))
  install.packages("caret", repos = "http://cran.us.r-project.org")
#if (!require(grid))
#  install.packages("grid", repos = "http://cran.us.r-project.org")
if (!require(gridExtra))
  install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if (!require(RColorBrewer))
  install.packages("RColorBrewer", repos = "http://cran.us.r-project.org")
if (!require(corrplot))
  install.packages("corrplot", repos = "http://cran.us.r-project.org")

## Change this path if the project is downloaded to another location.
proj_path <- "~/projects/attrition-and-performance"

## Variable for factor creation
levels <- c("Low","Medium","High","Very High")

## Reading table and storing in object ibm
data <-
  read.csv(paste(
    proj_path,
    "data/WA_Fn-UseC_-HR-Employee-Attrition.csv",
    sep = "/"
  ))

## Preparing, converting to factors and converting to numeric booleans fields
data <- data %>% transmute(age = Age,
                       #attrition = ifelse(Attrition == "Yes", 1, 0),
                       attrition = factor(Attrition, labels = c("Yes","No")),
                       travel = factor(BusinessTravel, levels = c("Non-Travel",
                                                                  "Travel_Rarely",
                                                                  "Travel_Frequently")),
                       employeeNumber = EmployeeNumber,
                       employeeCount = EmployeeCount,
                       dailyRate = DailyRate,
                       department = recode_factor(Department, "Human Resources" = "HR",
                                                  "Research & Development" = "R&D"),
                       distHome = DistanceFromHome,
                       education = factor(Education, labels = c("Below College",
                                                                "College",
                                                                "Bachelor",
                                                                "Master",
                                                                "Doctor")),
                       educationField = recode_factor(EducationField, "Human Resources" = "HR",
                                                      "Marketing" = "Mkt",
                                                      "Life Sciences" = "Life Sci.",
                                                      "Medical" = "MD",
                                                      "Technical Degree" = "Technical D."),
                       envSatisfaction = factor(EnvironmentSatisfaction, labels = levels),
                       gender = factor(Gender, labels = c("Female","Male")),
                       hourlyRate = HourlyRate,
                       jobInvolvement = factor(JobInvolvement, labels = levels),
                       jobLevel = JobLevel,
                       jobRole = JobRole,
                       jobSatisfaction = factor(JobSatisfaction, labels = levels),
                       maritalStatus = MaritalStatus,
                       monthlyIncome = MonthlyIncome,
                       monthlyRate = MonthlyRate,
                       numCompaniesWorked = NumCompaniesWorked,
                       over18 = ifelse(Over18 == "Y", 1, 0),
                       overTime = ifelse(OverTime == "Yes", 1, 0),
                       percentSalaryHike = PercentSalaryHike,
                       performance = factor(PerformanceRating, labels = c("Excellent",
                                                                          "Outstanding")),
                       performance = factor(performance, levels = c("Low",
                                                                    "Good",
                                                                    "Excellent",
                                                                    "Outstanding")),
                       relSatisfaction = factor(RelationshipSatisfaction, labels = levels),
                       standardHours = StandardHours,
                       stocOptionLevel = StockOptionLevel,
                       totalWkgYears = TotalWorkingYears,
                       trainingTimesLY = TrainingTimesLastYear,
                       workLifeBalance = factor(WorkLifeBalance, labels = c("Bad",
                                                                            "Good",
                                                                            "Better",
                                                                            "Best")),
                       yearsAtCompany = YearsAtCompany,
                       yearsInRole = YearsInCurrentRole,
                       yearsSinceProm = YearsSinceLastPromotion,
                       yearsWithManager = YearsWithCurrManager
                       )

## Identifying zero variability variables
zeroVar <- t(head(data[,nearZeroVar(data)], 1))
colnames(zeroVar) <- "Unique Value"

## Setting final dataset after removing zero variability variables
ibm <- data[,-nearZeroVar(data)]

## Delete variables that will no longer be used by Rmd
rm(list = c("levels"))

## Setting the color brewer palette for use in all plots
th <- scale_fill_brewer(palette = "Dark2")

## Create function for creating bar plots for categorical variables
catPlot <- function(data, columnName, xLab) {
  data %>% 
    ggplot(aes(get(columnName), fill = get(columnName))) +
    th +
    ylab("") +
    xlab(xLab) +
    geom_bar(show.legend = FALSE)
}

## Create function for creating density plots for continous variables
densPlot <- function(data, columnName, xLab) {
  data %>% 
    ggplot(aes(get(columnName), fill = "")) +
    th +
    ylab("") +
    xlab(xLab) +
    geom_density(show.legend = FALSE)
}

## Create function for correlation plots
corVariables <- function(data, xVar, yVar, xLab, yLab) {
  data %>%
    ggplot(aes(x = get(xVar), y = get(yVar), col = TRUE, fill = TRUE)) + 
    geom_smooth(method = "loess", show.legend = FALSE) +
    ylab(yLab) +
    xlab(xLab) +
    th
}

## Create a vector with the correlated variables to be removed:
vRmByCorr <- c("employeeNumber","performance","jobLevel","yearsAtCompany","yearsInRole","yearsWithManager","totalWkgYears","age") 

## Remove correlated variables from dataset ibm
ibm <- ibm %>% select(-vRmByCorr)

## Create data partition using 20% of data for test set
set.seed(13)
test_index <- createDataPartition(ibm$attrition, times = 1, p = 0.2, list = FALSE)
test <- ibm[test_index,]
train <- ibm[-test_index,]

## Models that will be used for ensemble
models <- c("glm", "lda",  "naive_bayes",  "svmLinear", 
            "knn", "rf", "ranger", "Rborist", "pcaNNet",
            "monmlp", "kknn", "mlp", "wsrf",
            #"adaboost",
            "gbm", "xgbTree", "rpart",
            "svmRadial", "svmRadialCost", "svmRadialSigma")

## Creating and training fits for each model
fits <- lapply(models, function(model){ 
  train(attrition ~ ., method = model, data = train)
})
names(fits) <- models

## Generate the matrix of predictions for the test set and get its dimensions
pred <- sapply(fits, function(object)
  predict(object, newdata = test))

#Accuracy for each model in the test set and the mean accuracy across all models
acc <- colMeans(pred == test$attrition)
all_acc <- mean(acc)

#Ensemble prediction building
votes <- rowMeans(pred == "Yes")
y_hat <- ifelse(votes > 0.5, "Yes", "No")
ens_acc <- mean(y_hat == test$attrition)

#Comparison of the individual methods to the ensemble
ind <- acc > mean(y_hat == test$attrition)
s_ind <- sum(ind)
m_ind <- models[ind]

#Calculate the mean accuracy of the new estimates
acc_hat <- sapply(fits, function(fit) min(fit$results$Accuracy))
ne_acc <- mean(acc_hat)

#Ensemble prediction building:
ind <- acc_hat >= 0.85
votes <- rowMeans(pred[,ind] == "Yes")
y_hat <- ifelse(votes >= 0.5, "Yes", "No")
final_acc <- mean(y_hat == test$attrition)