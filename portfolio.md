# Projects 1, 2 and Final Project

## [Project 1](project.1.2.ipynb) click to see code work
#### Written portion 
How does the time young adults spend on social media affect their mental health
I think this question is relevant because with the current use of social media like instagream, twitter and tik tok. The mental health repercussions are still not fully understood. I know multiple people that log into their phones and become jealous of the lives people show on social media or learn about news that makes them increasingly sad. So to understand this I want to use this data set to see the correlation between social media use and mood disorders
I think people that would care about this are parents on the fence about if they should let their kids get social media, maybe the companies of these social media outlets seeing if they should put age restriction on who can be on the app, or even health care providers to see if recommending getting off social media would be good for their patients

I got the data set form statistica.com
Each row represents a singular persona and their rating related to the question on the columns. Some of the key columns that I focused on are the mood disorder column to see the correlations between that and the rest of the time people spent on social media. I also used the headache and eye strain to see if social media use happened to affect people's physical health too. The data set is 513 by 32. Something that is missing in the data set are the ages of people so I’m assuming that all people giving their data are over the age of 18

I deleted the column that asked people how many hours they think they spend on social media and kept the actual amount of time they spend on social media. I think they were necessary because the data became too redundant and it leads to inaccurate answers. The more accurate answer would come from the information of the time they actually spend on social media. It will also lead to a better conclusion on how their physical and mental health correlate to the time spent on any outlet. I assumed that even if there was a zero in the entry it still held value to the data and chose to keep it. I thought about if I took away the “think_time” of the data would that actually have a big impact on the mental health of individuals but in the end I decided that could be formed as its own question and I wanted to focus on more fact based research/data

I created a bar chart, heat map, and scatter plot. I chose to make a heatmap because I had multiple variables and wanted to see if one had more of a connection than others, a scatter plot to see all the variables next to each other and look at the correlation and I made a face grid to just look at data in depth and understand what else I could do to it. I was interesting to see that linkin was connected to any type of mood disorder or mental health problem. It really helps me see the correlation in a visual way so it's easier to compare between variables and it also makes telling others what is happening much easier.

Brief interpretations of the existing plots 
Heatmap:
- Shows pairwise linear correlations between numeric variables.
- Look for large positive/negative cells (darker colors) to identify strong relationships to mood_disorder, time_loss_social_media, etc.

Scatter: time_loss_social_media vs nervous_anxious
- Shows point-level relationship and spread.
- Check for an upward trend (more time loss → higher nervousness/anxiety) and any outliers.

Scatter: social_media_before_bed vs mood_disorder (sized by time_loss_social_media)
- X = bedtime social media level, Y = mood disorder; marker size = time lost.
- Larger markers at higher Y or X indicate heavy users with worse mood; overlap means weak separation.

FacetGrid per app (KDEs of mood_disorder by usage level):
- Each row = one app; curves show mood_disorder distribution for usage levels.
- If higher-usage curves shift right or have a different shape, that suggests higher usage is associated with higher mood_disorder for that app.

Combined FacetGrid (all apps):
- Enables comparison across apps; look for which apps show the strongest distribution shifts by usage.
- Wide overlap across usage levels implies little difference; clear shifts indicate stronger association.

Looking at the visualization we can see that the amount of time a user spend on an app is directly related to a mood disorder, in all of the visualizations the more time a person spends the higher the anxiety, headache/eyestrain a person will have which also heightens their mood disorder. While the data does support that the more time an individual spends on social media it shows strong ties to a mood disorder but we also need to understand this is not concrete evidence because we are talking into account if the individual already has preexisting conditions. So while these conclusions can support that young adults should probably spend less time because it causes them to have a mood disorder it may not be the case for everyone and it needs to be studied in a wider aspect

Some bias could be the self sampling bias which could be different that the sample taken form the population. Self samplers could be more concerned with mental health and could systemically differ from the population. If I had more time with this project I would gather more data sets from Australia so I could compare the ban of social media for the data of how social media affects people under that age of 16 and would also change my data set to a group of people under 16.

#### References 
- Statistica.com for data set source
- Seaborn.py.data.org to see more graph references and to see the the sample python code to understand if I could do it with my data set
- youtube.com for tutorials on how to make specific graphs and what to see examples of people cleaning data sets
- copilot to help fix bugs in my code or fix syntax errors

  
-------



# Project 2
## [Project 2](project.2.2.ipynb) Click to see code
## Written Portion

### Predicting the reach of Wildfires
Every summer, across the forests of the United States, a familiar and frightening sight appears, the rising wildfires throughout the country start. For emergency responders, the most critical question isn't just "where is the fire?" but rather how big will it get, and how can we get it under control the fastest.
In my project I try to answer this question by investigating whether we can use fire's core size(Area) to accurately predict its boundaries(Perimeter). Hopefully being able to give valuable insight on where to dig firebreaks and which communities to evacuate. 


#### The problem: Unpredictable Flames
A wildfire isn't a perfect circle. It is shaped by the wind, terrain, and fuel given to it. While we often hear about the acreage of a fire, the perimeter is actually where the most important part of the fire takes place, it helps first responders know where to be to stop the further spread by removing vegetation or spray chemical agents. 
My research question was simple: can a machine learning model look at the footprint of past fires and predict the length of its edge(the active burning perimeter)? If successful, this could give first responders a faster way to estimate the resources needed to contain a blaze


#### Investigating Data
Using the environmental data that I got from Data.gov on wildfires from a federal source I analyzed fire events, however with all the noisy data some cleaning needed to be done first. I had a couple of challenges with this data set one being how there was a lot of repetition throughout the whole data set. I discovered including all the data from the data set was causing multicollinearity meaning it essentially stopped understanding the data and rather just doing unit conversions. On top of that I removed ghost data points such as negative perimeters of lengths, and duplicate info to ensure that focus was put on the physics of the fire.


#### Why one tool isn't enough
I tested three different types of models to predict the fires edge:
1. Linear Regression: Great standard, predictable fires but struggles with complexity. I used this model to be a base control group. If the prediction could come out accurate with a simple straight line then no need for more complex measures but that was not the case. I also assumed with would be the best with my data because bigger fire equals bigger perimeters and bigger damage.
2. Regularized Models(Ridge and lasso): These acted as a filter, ignoring the less important data to focus on the strongest predictors. My data set had multicollinearity and these Ridge and Lasso add a penalty for complexity, forcing the model to only focus on the most important variables. This prevents the model from "cheating" by using two different area measurements to memorize specific fires, forcing it to find a general rule that applies to all fires.
3. Decision Trees: These were able to handle the non-linear nature of wildfires recognizing that a small fire in a windy canyon behaves differently than a small fire on a flat plain. This model can categorize fires into different types and apply different logic to each. Helping further all the models to push a better outcome.

#### Visualizations
I know just talking about subjects can get confusing and boring, I wanted to share some images from the research that I think would build a better picture to truly understand correlation.

These are heat maps
<img width="916" height="741" alt="Screenshot 2026-04-05 224405" src="https://github.com/user-attachments/assets/6cc1237d-d4e5-44f1-9778-a16786479ce0" />

This image is one the first images I made to visualize the correlation between all the variables and this is the heat map that I ended with after all my cleaning

<img width="983" height="850" alt="Screenshot 2026-04-05 224349" src="https://github.com/user-attachments/assets/8908c48e-7ac1-484b-a06c-8c410f99c61a" />

In short I calculated the R^2 and an RMSE with these pictures. It was successfully telling us that we can estimate a fire boundary, but it also showed that shape complexity was my model's biggest challenge. For large fires we would still need human double check

This is a Tree model!
<img width="1733" height="965" alt="Screenshot 2026-04-05 224336" src="https://github.com/user-attachments/assets/650741cb-deaf-4a84-9bd7-6fb697e79ff7" />

This shows the essential data preprocessing phase by cleaning and preparing your wildfire dataset for machine learning. First, it identifies and removes multicollinearity by dropping redundant columns, ensuring the model doesn't get lazy by training on duplicate area metrics. Next it handles missing values through imputation, filling in gaps so the algorithms have a complete dataset to analyze. It filters out outliers and invalid data. Finally the code organizes the remaining features into a clean format that is ready for any model.

Finally pair plots
<img width="2211" height="2211" alt="output first" src="https://github.com/user-attachments/assets/784fd919-f47d-4644-bd6e-aed251dcd67d" />


<img width="2211" height="2211" alt="output" src="https://github.com/user-attachments/assets/05556495-ddd6-40a3-8edc-f172697af809" />

The primary difference between these two pictures is how they handle the relationship between a fire's area and its perimeter. The linear Reaction Model(Top) assumes a fixed, straight line relationship, which results in a rigid "best fit" that struggles to account for unique, jagged shapes of large wildfires. In contrast the Decision tree model(Bottom) uses a flexible, non-linear approach that can better adapt to complex  branching behavior of fire boundaries, leading to tighter clusters of predictions. While both models show uncertainty as the fire grows in size the Decision tree pair plots ability to categorize different fire types allow for a more nuanced and accurate estimation of invisible edges.



#### The Impact
The most eye opening moment of this research/data investigation came when I realized that my model was most vulnerable when the fire was the most irregular. For a perfect circular fire, the prediction was easy. However, for jaded fires, ones that have multiple factors impacting it, the ones hardest to contain, the model's errors skyrocketed and it shows how much more work this model needs.


#### References 
- Professors Dr. Blekking and Dr.Bennedict on their help and advise on which data sets to use
- Data.gov for data set source
- Seaborn.py.data.org, panda.pydata.org, numpy.org to see more graph references and to see the the sample python code to understand if I could do it with my data set
- youtube.com for tutorials on how to make specific graphs and what to see examples of people cleaning data sets
- Google.com the search up questions about fires and understand each variable given
- copilot to help fix bugs in my code or fix syntax errors

-------
# Fianl Project
## [Final Project]() Click to see code
## Written Portion
#### Introduction 
Crude oil is one of the most critical resources in the global economy, influencing transportation, energy production, and overall economic stability. Because oil is traded globally, even small changes in supply, demand, or geopolitical conditions can significantly impact prices and availability. 

This project investigates the relationship between U.S. crude oil imports and global oil prices. Specifically, it explores whether import patterns such as where oil comes from, how much is imported, and what type of crude is received have meaningful connection to changes in spot prices for WTI and Brent crude oil.

Understanding this relationship is important because oil prices directly affect consumers through gas prices, inflation, and energy costs. While imports reflect the stability and structure of global supply chains. 	

#### Data
This project combines two real world datasets from the U.S. Energy Information Administration (EIA) and related energy market sources. 

1) U.S. Crude Oil Imports (EIA API Dataset)
This dataset contains monthly records of crude oil imports into the United States, 
Including:
originName: Country of origin (e.g Canada, Mexico, Saudi Arabia)
destinationName: Refinery or port destination
gradeName: Type of crude oil 
quantity: Volume of crude oil imported 
period: Monthly time period of imports
	This dataset helps explain how oil physically enters the U.S. supply chain. 

2) Crude oil Spot Prices Dataset 
This dataset contains historical data spot prices for:
WTI Crude Oil
Brent Crude Oil
It includes daily price values in U.S. dollars per barrel over time, capturing fluctuations 
caused by global supply and demand changes, geopolitical events, and market speculation.

#### Preprocessing and Exploratory Data Analysis
The dataset was cleaned and prepared for analysis by standardizing formats, handling missing values, and converting date fields into consistent time based formats to allow merging.

The imports dataset was grouped and analyzed by country of origin to identify major suppliers. The results showed that Canada, Non OPEC countries, and Middle Eastern nations were among the largest contributors to U.S. crude oil imports.
The spot price database was filtered to focus only on crude oil related values and converted into a time series format. Initial exploratory analysis showed clear vitality in both WTI and Brent consistently trading at a higher value. 

Additional exploratory analysis included:
Scatterplots comparing import quantities across origins and crude grades
This displays the relationship between import quantity, origin, and crude oil grade across the top origins in the dataset. The distribution of points highlights variations in quantity by region and grade type, helping identify trends, clusters, and potential outliers in import activity.
Time series visualizations of oil price fluctuations
This shows changes in WTI Crude Oil and UK Brent Crude Oil spot prices over time. The visualization highlights price trends, fluctuations, and major spikes or drops in the market across different years.
Heatmaps showing distribution patterns across time and categories
This shows the correlation between variables in the merged imports and spot prices dataset. The colors represent the strength of relationships between variables, with most values showing weak correlations. This suggests that the variables are largely independent and do not strongly influence one another.
These visualizations helped reveal that oil imports are relatively stable compared to volatile price movements. 

#### Modeling
To investigate whether crude oil imports could help explain or predict oil prices, multiple machine learning models were applied to the merged dataset.

The dataset included features such as: 
Import quantity
Origin country 
Crude oil grade
Destination type

The target variable was the spot price (WTI and Brent values).

Three modeling approaches were used: 
Decision Tree Regressor 
	A decision tree model was trained with limited depth to prevent overfitting. However, the 
model showed very weak predictive performance with R² score score close to zero. 
Random Forest Regressor
A more advanced ensemble model was applied to improve predictive performance. While it performed slightly better than the decision tree, the R² score remained low, indicating limited explanatory power from the available features.
Logistic Regression
	To further test predictability, oil prices were converted into a binary classification 
problem. Logistic regression was used, but the model performed close to random 
guessing, indicating weak signal strength in the input features. 

	Cross validation was also applied, and results consistently showed low predictive 
performance across all models. 

#### Results
The key finding from this project is that crude oil import characteristics do not strongly predict or explain crude oil spot prices.
Decision Tree R² ≈ 0.01 (very weak performance)
Random Forest R² ≈ 0.12 (slightly improved but still weak)
Logistic Regression accuracy ≈ 50% (equivalent to random chance)

Correlation analysis between import quantity and oil prices showed near zero relationships, confirming that no strong linear connection exists between the variables. 

These results suggest that oil prices are influenced more by macroeconomic and global market factors than by direct import volumes or supply chain structure. 

#### Key insights
The U.S. relies heavily on a few key import partners, especially Canada.
Brent crude consistently trades at higher prices than WTI. 
Oil imports remain relatively stable over time compared to volatile price movements. 
Import patterns alone are not sufficient to predict oil prices. 

#### Conclusion
This project demonstrates that while crude oil imports are essential for maintaining U.S. energy supply, they do not directly determine oil prices. 

Instead, oil prices are shaped by broader global forces such as geopolitical events, production decisions by major oil producing countries, and shifts in global demand. 

In contests, imports reflect physical supply chains, while prices reflect global financial market behavior. These two systems operate in parallel but are not strongly dependent on each other in a predictive sense. 

Overall, this analysis highlights the complexity of global energy systems and the limitations of using supply side data alone to forecast market price. 

#### Resources
U.S. Energy Information Administration (EIA). Crude Oil Imports Data API. https://www.eia.gov/


U.S. Energy Information Administration (EIA). Petroleum & Other Liquids Data. Available at: https://www.eia.gov/petroleum/data.php

AI Acknowledgement: ChatFPT was used to assist in data cleaning. 
U.S. Energy Information Administration (EIA)

S. Petroleum & Other Liquids Spot Price Data

