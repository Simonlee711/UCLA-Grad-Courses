
# Load libraries
library(tidyverse)
library(scales)

# Set working drive (note, you must change the path name to your directory)
setwd("/Users/simonlee/MGMT298/")

# Publicly available data from GitHub https://github.com/owid/covid-19-data/tree/master/public/data/vaccinations

###################################
# US Vaccinations by Manufacturer #
###################################

# Import data and store in dataframe called MANUFACTURER
# Can also manually import using Import Dataset on the right panel -->
MANUFACTURER = read.csv("vaccinations-by-manufacturer.csv")

# Format date column
MANUFACTURER$date = as.Date(MANUFACTURER$date, "%Y-%m-%d")

# Filter to include only US data (PIPE OPERATOR IN DPLYR)
US = MANUFACTURER %>%
  filter(location == "United States")

# Set axis limits
min = as.Date("2021-01-01")
max = as.Date("2021-04-01")

# Plot cumulative vaccinations by date (aes= aesthetics)
ggplot(data=US, aes(x=date, y=total_vaccinations, color=vaccine)) + 
  geom_line() 

# Improved plot
ggplot(data=US, aes(x=date, y=total_vaccinations, color=vaccine, shape=vaccine)) + 
  geom_line(size=0.5) + geom_point(size=1) +
  scale_x_date("", date_labels = "%b-%d", limits = c(min, max)) + 
  scale_y_continuous("Cumulative doses", breaks = seq(0, 80000000, 10000000), labels = comma) +
  scale_color_discrete("Manufacturer") + scale_shape_discrete("Manufacturer") +
  ggtitle("COVID-19 Vaccination Doses in US by Manufacturer, as of March 30, 2021")  


############################
# US Vaccinations by State #
############################

# Import data and store in dataframe called STATES
STATES = read.csv("us_state_vaccinations.csv")

# Format date column
STATES$date = as.Date(STATES$date, "%Y-%m-%d")

# Filter data to keep only the most recent date and remove aggregate value for United States
LATEST = STATES %>%
  filter(date == max(STATES$date) & location != "United States")
         
# Plot cumulative vaccinations by state
ggplot(data=LATEST, aes(x=location, y=total_vaccinations)) +
  geom_col() 

# Improved plot
ggplot(data=LATEST, aes(x=reorder(location, total_vaccinations), y=total_vaccinations)) + 
  geom_col(fill = "Blue") + coord_flip() +
  scale_x_discrete("Location") + 
  scale_y_continuous("Cumulative doses", breaks = seq(0, 20000000, 5000000), labels = comma) +
  ggtitle("COVID-19 Vaccination Doses by US State, as of March 30, 2021")

# Plot percent vaccinated by state
ggplot(data=LATEST, aes(x=reorder(location, people_vaccinated_per_hundred), y=people_vaccinated_per_hundred)) + 
  geom_col(fill = "Red") + coord_flip() +
  scale_x_discrete("Location") +
  scale_y_continuous("Percent of population with at least 1 dose") + 
  ggtitle("COVID-19 Vaccination Coverage by US State, as of March 30, 2021")
  
# Plot 3 variables
ggplot(data=LATEST, aes(x=people_vaccinated_per_hundred, y=share_doses_used, size=total_vaccinations, label=location)) +
  geom_point() + geom_text()  

# Improved plot   https://github.com/aljrico/gameofthrones
library(ggrepel)
library(gameofthrones)
ggplot(data=LATEST, aes(x=people_vaccinated_per_hundred, y=share_doses_used, size=total_vaccinations, color=total_vaccinations, label=location)) +
  geom_point(alpha=0.5) + geom_text_repel(size=3, max.overlaps = Inf) + 
  scale_x_continuous("Percent of population with at least 1 dose") + 
  scale_y_continuous("Share of allocated doses used") +
  scale_size("Cumulative doses", range=c(0.5, 10), labels=comma) +
  scale_color_got("Cumulative doses", discrete=FALSE, option="Lannister", labels=comma) + theme_bw() +
  ggtitle("COVID-19 Vaccinations by US State, as of March 30, 2021") 

# Animated plot for select states  
# https://www.datanovia.com/en/blog/gganimate-how-to-create-plots-with-beautiful-animation-in-r/
library(gganimate)
library(gifski)

SELECT = STATES %>%
  filter(location == "California" | location == "New York State" | location == "Florida" | location == "Texas")

plot = ggplot(data=SELECT, aes(x=people_vaccinated_per_hundred, y=share_doses_used, size=total_vaccinations, color=location)) +
  geom_point() +
  scale_size("Cumulative doses", range=c(0.5, 10), labels=comma) +
  labs(title="{closest_state}", x="Percent of population with at least 1 dose", y="Share of allocated doses used") +
  transition_states(date) + ease_aes("linear") + shadow_mark(alpha=0.1)

animate(plot, nframes = 200, renderer = gifski_renderer("gganim.gif"))
