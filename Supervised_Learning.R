#Setting Working Directory
getwd()
setwd("C:/Users/SaiSpandana/OneDrive/Desktop/Supervised_Learning_Project")

#Loding packages
library(readr)  #Loading the data
library(dplyr) # library for data manipulation
library(tidyr) 
library(glue)
library(ggplot2)
library(plotly)
library(GGally)
library(rsample)
library(MASS)
library(performance)
library(lmtest)
library(car)
library(gtools)
library(caret)
library(class)
library(e1071)
library(ROCR)
library(partykit)
library(gridExtra)
library(randomForest) 
#Loding the Data
wine <- read.table("winequality-red.csv", sep = ";", header = TRUE)
view(wine)

#Dataset Insights
colnames(wine)

#Feature of data
head(wine)

#Dimension of data
dim(wine)

#Structure of data
str(wine)

#Summary of data
summary(wine)

#Checking for the null values
apply(wine, 2, function(x)sum(is.na(x)))

# Unique valules in the target variable
prop.table(table(wine$quality))

# Draw a histogram for a given dataframe and variable
# Use deparse() and substitute() functions to decode column name from
# a variable passed as an argument to the function, to be displayed
# on x axis (xlab())
draw_hist <- function(dataframe, variable)
{
  # Save histogram definition to the plot variable
  plot <- ggplot(data = dataframe, aes(x = variable)) + 
    geom_histogram(color = 'black', fill = '#099DD9') +
    xlab(deparse(substitute(variable)))
  return(plot)
}
# Build a matrix of small histograms with 3 columns
# using customly defined draw_hist() function
grid.arrange(draw_hist(wine, wine$fixed.acidity),
             draw_hist(wine, wine$volatile.acidity),
             draw_hist(wine, wine$citric.acid),
             draw_hist(wine, wine$residual.sugar),
             draw_hist(wine, wine$chlorides),
             draw_hist(wine, wine$free.sulfur.dioxide),
             draw_hist(wine, wine$total.sulfur.dioxide),
             draw_hist(wine, wine$density),
             draw_hist(wine, wine$pH),
             draw_hist(wine, wine$sulphates),
             draw_hist(wine, wine$alcohol),
             draw_hist(wine, wine$quality),
             ncol = 3)

#Visuvalizing the dependent variable
# Plot a histogram of quality values
ggplot(data = wine, aes(x = quality)) +
  geom_histogram(color = 'black', fill = 'deepskyblue1', binwidth = 1) +
  # Used to show 0-10 range, even if there are no values close to 0 or 10
  scale_x_continuous(limits = c(0, 10), breaks = seq(0, 10, 1)) +
  xlab('Quality of Red Wine') +
  ylab('Number of Red Wines')

wine$quality_high <- as.factor(ifelse(wine$quality>=6, 1, 0))
glimpse(wine$quality_high)
prop.table(table(wine$quality_high))

#Cross validation 
#Splitting the data
RNGkind(sample.kind = "Rounding")
set.seed(123) 

# index sampling
index_wine <- initial_split(wine, prop = 0.8, strata = "quality_high")

# splitting
wine_train <- training(index_wine)
wine_test <- testing(index_wine)

#checking proportions on separated dataframes
prop.table(table(wine_train$quality_high))
prop.table(table(wine_test$quality_high))

#Exploratory data analysis and visualization
#box plot of variables
p2 <- ggplot(data = stack(wine %>% dplyr::select(volatile.acidity, citric.acid, chlorides, sulphates)), mapping = aes(x = ind, y = values)) +
  geom_boxplot(fill = "pink")+
  theme_dark()+
  labs(title = "Boxplot of Volatile Acidity, Citric Acid, Chlorides, Sulphates",
       x = "Predictors",
       y = "Value")

ggplotly(p2)

p3 <- ggplot(data = stack(wine %>% dplyr::select(fixed.acidity, residual.sugar, alcohol)), mapping = aes(x = ind, y = values)) +
  geom_boxplot(fill = "green")+
  theme_dark()+
  labs(title = "Boxplot of Fixed Acidity, Residual Sugar, Alcohol",
       x = "Predictors",
       y = "Value")

ggplotly(p3)

p4 <- ggplot(data = stack(wine %>% dplyr::select(free.sulfur.dioxide, total.sulfur.dioxide)), mapping = aes(x = ind, y = values)) +
  geom_boxplot(fill = "cyan")+
  theme_dark()+
  labs(title = "Boxplot of Free and Total Sulfur Dioxide",
       x = "Predictors",
       y = "Value")

ggplotly(p4)

#Checking correlations between predictors
ggcorr(wine_train, label = T, hjust = 0.9, label_size = 3, layout.exp = 3)

#By observing the plot there is a strong correlation b/w free.sulfur.dioxide,total.sulfur.dioxide, fixed.acidity
# volatile.acidity

#Scatter plots for strong correlated variables
#Free.sulfur.dioxide and Total.sulfur.dioxide
p5 <- ggplot(data = wine %>% mutate(label = glue("Free Sulfur Dioxide = {free.sulfur.dioxide} ppm,
                                           Total Sulfur Dioxide = {total.sulfur.dioxide} ppm,
                                           Ratio = {round(free.sulfur.dioxide/total.sulfur.dioxide,3)}")),
                 mapping = aes(x = free.sulfur.dioxide, y = total.sulfur.dioxide, text = label))+
  geom_point(aes(color = free.sulfur.dioxide/total.sulfur.dioxide))+
  theme_dark()+
  labs(x = "Free Sulfur Dioxide (ppm)",
       y = "Total Sulfur Dioxide (ppm)",
       color = "Ratio of Free:Total")

ggplotly(p5, tooltip = "label")

#Fixed Acidity and Volatile Acidity
p6 <- ggplot(data = wine %>% mutate(label = glue("Fixed Acidity = {fixed.acidity},
                                                      Volatile Acidity = {volatile.acidity},
                                                      Ratio = {round(fixed.acidity/volatile.acidity,3)}")),
                 mapping = aes(x = fixed.acidity, y = volatile.acidity, text = label))+
  geom_point(aes(color = fixed.acidity/volatile.acidity))+
  theme_dark()+
  labs(x = "Fixed Acidity",
       y = "Volatile Acidity",
       color = "Ratio of Fixed:Volatile")

ggplotly(p6, tooltip = "label")

#Fixed Acidity and pH
p7 <- ggplot(data = wine %>% mutate(label = glue("Fixed Acidity = {fixed.acidity},
                                                      pH = {pH}")),
                 mapping = aes(x = fixed.acidity, y = pH, text = label))+
  geom_point(color = "aquamarine")+
  theme_dark()+
  labs(x = "Fixed Acidity",
       y = "pH")

ggplotly(p7, tooltip = "label")

#Volatile Acidity and Citric Acid
p8 <- ggplot(data = wine %>% mutate(label = glue("Volatile Acidity = {volatile.acidity},
                                                      Citric Acid = {citric.acid}")),
                 mapping = aes(x = volatile.acidity, y = citric.acid, text = label))+
  geom_point(color = "DarkGreen")+
  theme_dark()+
  labs(x = "Volatile Acidity",
       y = "Citric Acid")

ggplotly(p8, tooltip = "label")

table(wine$quality_high)

#Finding the means
wine_char <- wine %>% 
  mutate(quality_high = ifelse(quality>=6, "high", "low")) %>% 
  group_by(quality_high) %>% 
  summarise_all(mean) %>% 
  dplyr::select(-quality)

wine_char

p9 <- wine_char %>% 
  pivot_longer(cols = -quality_high, names_to = "names", values_to = "values") %>% 
  mutate(label = glue("Red Wine Quality? {quality_high}
                      Average of {names} = {round(values,2)}")) %>% 
  ggplot(mapping = aes(x=names, y=values))+
  geom_line(aes(group = quality_high, color = quality_high))+
  geom_jitter(mapping = aes(x=names, y=values, color = quality_high, text = label))+
  theme_dark()+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
  labs(title="Average Characteristics of Top-Rated vs Lower-Rated Red Wines",
       x="Predictor",
       y="Value",
       color="Red Wine Quality?")

ggplotly(p9, tooltip = "label")

#Classification Model: k-NN
#Pre-processing of data
summary(wine_train)

#Separating Predictors and Target Variables
wine_train_x <- wine_train %>% 
  dplyr::select(-c("quality", "quality_high"))

wine_train_y <- wine_train %>% 
  pull(quality_high)

wine_test_x <- wine_test %>% 
  dplyr::select(-c("quality", "quality_high"))

wine_test_y <- wine_test %>% 
  pull(quality_high)

#Scaling the data
wine_train_x_scaled <- scale(wine_train_x)
wine_test_x_scaled <- scale(wine_test_x,
                            center = attr(wine_train_x_scaled, "scaled:center"),
                            scale = attr(wine_train_x_scaled, "scaled:scale"))
summary(wine_train_x_scaled)
summary(wine_test_x_scaled)

#Finding Optimum k
sqrt(nrow(wine_test_x_scaled))

#Fitting the model
wine_knn_pred_k17 <- knn(train = wine_train_x_scaled,
                         test = wine_test_x_scaled,
                         cl = wine_train_y,
                         k=17)

wine_knn_pred_k15 <- knn(train = wine_train_x_scaled,
                         test = wine_test_x_scaled,
                         cl = wine_train_y,
                         k=15)

wine_knn_pred_k19 <- knn(train = wine_train_x_scaled,
                         test = wine_test_x_scaled,
                         cl = wine_train_y,
                         k=19)

#Model evaluation
#Using k=17 (closest number to optimun)
confusionMatrix(data = wine_knn_pred_k17,
                reference = wine_test_y,
                positive = "1")
#Using k=15 (closest number to optimun)
confusionMatrix(data = wine_knn_pred_k15,
                reference = wine_test_y,
                positive = "1")
#Using k=19 (closest number to optimun)
confusionMatrix(data = wine_knn_pred_k19,
                reference = wine_test_y,
                positive = "1")

#Model fitting of Random forest
#Fitting the regression model of Random forest
 set.seed(314)
 
ctrl <- trainControl(method = "repeatedcv",
                      number = 5, # k-fold
                      repeats = 3) # repetition
 
wine_forest_reg <- train(quality ~ .,
                          data = wine_train %>% dplyr::select(-quality_high),
                          method = "rf", # random forest
                          trControl = ctrl)

saveRDS(wine_forest_reg, "wine_forest_reg.RDS") # saving model

#Fitting classification model
wine_forest_cla <- train(quality_high ~ .,
                         data = wine_train %>% dplyr::select(-quality),
                         method = "rf", # random forest
                         trControl = ctrl)
 
saveRDS(wine_forest_cla, "wine_forest_cla.RDS") # saving model

#Model Evaluation
wine_forest_reg <- readRDS("wine_forest_reg.RDS")
wine_forest_reg

wine_forest_reg$finalModel

#Classification model
wine_forest_cla <- readRDS("wine_forest_cla.RDS")
wine_forest_cla
wine_forest_cla$finalModel
#Confusion matrix
wine_forest_cla_pred <- predict(object = wine_forest_cla,
                                newdata = wine_test,
                                type = "raw")

confusionMatrix(data = wine_forest_cla_pred,
                reference = wine_test$quality_high,
                positive = "1")
