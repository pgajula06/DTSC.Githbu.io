# Projects

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
## [Project2]() Click to see code
### Written Portion
--------

### Predicitng the reach of Wildfires
Every summer, across the forests of the United States, a familia and frightening sight appears, the rising wildfires thorugh out the contries start. For emergency responders, the most critical question isn't just "where is the fire?" but rather how big will it get, and how can we get it under control the fastest.
In my project I try to take this question by invesitgating wheater we can use fire's core size(Area) to accurately perdict its boundaries(Perimeter). Hopefully being able to give valuable insight on where to dig firebreaks and which communties to evacuate. 

#### The problem: Uperdicatable Flames
A wildfire isn't a perfect circle. It is shaped by the wind, terrain, and fuel given to it. While we often hear about the acerage of a fire, the perimeter is actually where the most important part of the fire takes place, it helps first repsonders know where to be to stop the furthur spread by removing vegitation or spray chemical agents. 
My reasearch question was simple, can a machine learning model look at the footprint of past fires and perdict the length of its edge(the active burning perimeter)? If succuessful, this could give first responders a fater way to estimate the resources needed to contain a blaze

#### Invesitagting Data
Using the enviormental data the I got from Data.gov on wildfries from a federal source I analyzed fire events, however with all the noisy data some cleaning needed to be done first. I had a couple of challenges with this data set one being how there was a lot of repetition through out the whole data set. I discovered inculding all the data from the data set was causing multicollinerity meaning it essentailly stopped understanding the data and rather just doing unit conversions. On top of that I removed ghost data point such as negaitve perimeters of lengths, and ducplicate info to ensure that focues was put on the physics of the fire.

#### Why one tool isn't enough
I tested three different types of models to perdict the fires edge:
1. Linear Regression: Great standard, predictable fires but stuggles with complex. I used this model to be a base control group. If the perditionc could come out accuarte with a simple stright line than no need for more complex meassures but that was not the case. I also assumed with would be the best with my data becasue bigger fire equals biggere peimeters and bigger damage.
2. Regularized Models(Ridge and lasso): These acted as a filter, ignoring the less important data to focus on the strognest predictors. My data set had mulitcollinearity and these Ridge and Lasso add a penalty for complexity, forcing the mdoel to only focus on the most importnat variables. This prevents the model from "cheating" by using two different area measurements to memorize specific fire, forcing it to find a general rule thatapplies to all fires.
3. Decision Trees: These were ablt to handle the non-linear nature of wildfires recognizing that a small fire in a windy canyon behaves differently then a samll fire on a flate plain. This model can catogorize fires into different types and apply different logic to each. Helping further all the models to push a better outcome.

#### The Impact
The most eye opening moment of this research/data invesitagation came when I realized that my model was most vulnerable when the fire was the most irregular. For a perfect circular fire, the prediction was easy. However, for jadded fires ones that has multiple factors impacting it, the ones hardest to contain, the models errors skyroceted and it shows how much more work this model needs.

#### References 
- Professors Dr. Blekking and Dr.Bennedict on their help and advise on which data sets to use
- Data.gov for data set source
- Seaborn.py.data.org, panda.pydata.org, numpy.org to see more graph references and to see the the sample python code to understand if I could do it with my data set
- youtube.com for tutorials on how to make specific graphs and what to see examples of people cleaning data sets
- Google.com the search up questions about fires and understand each variable given
- copilot to help fix bugs in my code or fix syntax errors
