
# Load libraries
library(tidyverse)
library(scales)

# Set working drive (note, you must change the path name to your directory)
setwd("C:/Users/elong/Dropbox/UCLA/HealthcareAnalytics/2021 Spring/Week 1")

# Import data and store in dataframe called STATES
STATES = read.csv("us_state_vaccinations.csv")

STATE = STATES %>%
  filter(location != "United States" & date < "2021-02-01" & date != "2021-01-16" & date != "2021-01-17" &  date != "2021-01-18") 

ggplot(data = STATE, mapping = aes(x = date, y = people_vaccinated, colour = location)) + 
  geom_point() + 
  scale_x_discrete(guide = guide_axis(check.overlap = TRUE))


# Format date column
STATES$date = as.Date(STATES$date, "%Y-%m-%d")

SELECT = STATES %>%
  filter(location != "United States" 
         & location != "American Samoa" 
         & location != "Bureau of Prisons" 
         & location != "Dept of Defense" 
         & location != "Federated States of Micronesia" 
         & location != "Guam" 
         & location != "Indian Health Svc" 
         & location != "Long Term Care"
         & location != "Marshall Islands"
         & location != "Northern Mariana Islands"
         & location != "Puerto Rico"
         & location != "Republic of Palau"
         & location != "Veterans Health"
         & location != "Virgin Islands")

min = as.Date("2021-01-01")
max = as.Date("2021-04-01")

ggplot(data = SELECT, mapping = aes(x = date, y = people_vaccinated_per_hundred, color = people_vaccinated)) +
  facet_wrap(~location) +
  geom_col() + 
  scale_x_date("", date_labels = "%b", limits = c(min, max)) +
  scale_y_continuous("Vaccinations per 100 people") +
  scale_colour_continuous("Total vaccinated", type = "viridis", labels = comma)
