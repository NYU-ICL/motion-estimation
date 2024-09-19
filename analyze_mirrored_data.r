# Install necessary packages if not already installed
if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr")
}
if (!requireNamespace("readr", quietly = TRUE)) {
  install.packages("readr")
}

# Load the packages
library(dplyr)
library(readr)

# Read the CSV file into a data frame
data <- read_csv("./io/data/fullv2/agg/fullv2_all_unmirrored.csv")

# Create a new column for the sign of 'TrialConfig.MetaConfig.Condition.Scene.View.Heading'
data <- data %>%
  mutate(HeadingSign = case_when(
    TrialConfig.MetaConfig.Condition.Scene.View.Heading > 0 ~ "Positive",
    TrialConfig.MetaConfig.Condition.Scene.View.Heading < 0 ~ "Negative",
    TrialConfig.MetaConfig.Condition.Scene.View.Heading == 0 ~ "Zero"
  ))

# Perform one-way ANOVA
anova_result <- aov(TrialResponse.AgainstResponseProbability.mean ~ HeadingSign, data = data)

# Print the summary of the ANOVA
summary(anova_result)
