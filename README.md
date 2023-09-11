# Final Project: Neural Networks for Predicting Flood Density in New York State
* Full report in repo 

## Abstract / Summary
The risks of flash flooding are increasing in New York state, as climate and land use changes cause more severe storm events and greater likelihoods of flooding during storms.  These floods can cause disastrous loss of life and property that impacts communities long after the storm is over, yet forecasting flooding is difficult due to the large number of variables with non-linear relationships involved.  This is the motivation for using artificial neural networks to create accurate predictions of flooding in New York using publicly available data.  Mapping tools from ArcGIS online and Pro were used to assemble flood risk geologic and hydraulic variables, and flood density was used as a predictor for modeling flood risks.  Because of the infrequency of severe flooding events, the data has an inflated amount of zeros for flood density, which was overcome by scaling the data to a smaller range and resampling the lower range of zero risks to create less imbalanced data.  Fully connected neural networks of multi layer perceptrons were used to generate a regression of flood density.  Deeper networks with more hidden layers and trained on resampled data performed the best, leading us to conclude that resampling is a valid strategy for overcoming issues of studying natural weather phenomena with zero-inflated outcome variables.

Problem Statement: Apply artificial neural networks to predict flood density from publicly available datasets in New York state.

### Background and Significance
Climate change is shifting patterns in the frequency and intensity of natural disasters.  In the United States, extreme precipitation in the Northeast may increase by 52% by 2099 (Picard, 2023), resulting in increased risks of flood and flash flood events. In July 2023, severe flooding devastated several states along the east coast.  West Point in New York received over 7 inches of rain over the course of 4 hours, which shut down trains lines and interstates (Betts, 2023). On a personal level, the two authors of this project would have been caught in the storm in West Point, but changed plans last minute to come back to Maryland ahead of schedule and avoided the floods by lucky circumstances.
To focus the scope of the study, we propose the target state of New York.  According to the Risk Factor tool, over the next 30 years, New Yor has 17% of total state properties at over 26% of flood risk (Bates, 2021).  New York also has large swaths of farmland, with 6.9 million acres, as well as highly populated urban areas, which are greatly affected by flash floods. 
Flash floods have significant and damaging social and economic impacts, from loss of life to property damage that sets back communities for years afterwards.  Flood predictors can be utilized to build climate-resilient communities.  This includes land use and zoning planning considerations for local officials and natural disaster preparation such as asset prepositioning and response coordination practice.  The property insurance industry also requires accurate risk assessments, and homeowners can be properly prepared.	

### Motivation for Deep Learning
We have previously utilized non-deep learning methods to predict intermittent precipitation in Tacoma, Washington using data collected from weather services. Our previous methodology involved the use of a similar zero-inflated approach to modeling, using an XGBoost classifier to classify instances of precipitation occurring. We then trained a random forest regression model on all instances of precipitation that did occur. However, this methodology only achieved an accuracy of 70% when predicting the following hour’s precipitation in millimeters.
We believe that as extreme weather becomes increasingly common, it is critical to be able to accurately forecast these events for the sake of safety and public infrastructure. By using neural networks, we hope to be able to model the complex non-linear relationships to forecast upcoming events more accurately and produce maps of where flooding may take place. 

## Data Sources and Variables
### Data Sourcing and Loading
Public data sets available on ArcGIS online through a UMD academic license were utilized to build the variables of this project.  Historic flood data was sourced from NOAA’s Storm Events Database, which lists historic flood and flash flood events since the 1950s and associated characteristics (National Centers for Environmental Information).  The specific target variable is flood density, which was calculated in ArcGIS as the magnitude of flood occurrences per cell using historic flood locations in the Storm Events database.

Input variables came from geomorphologic data that are known to be associated with flood risks, such as elevation, slope, and precipitation. Information about land cover features, which can impact how quickly water is absorbed vs. run off during precipitation events, was sourced from the Multi-Resolution Land Cover Characteristics Consortium’s National Land Cover Database (Dewitz, 2019). Hydrologic soil group data, which divides soils into classes based on water infiltration potential, came from the USDA gridded soil survey geographic (GSSURGO) dataset (Soil Survey Staff).  All variables had a resolution of 30 meters, except annual precipitation (Fick, 2017), which was resampled at 30 meters from a 5 meter resolution.  After removing null values, the final dataset contained 9 million rows.  A detailed table of input and target variables and their characteristics can be found in Table 1.

* Table 1

  | Variable           | Description  | Values                 | Source                |  
  |--------------------|--------------|------------------------|-----------------------|  
  | x                  | Pixel Coord  | Lat                    | ArcGIS Online         |  
  | y                  | Pixel Coord  | Lon                    | ArcGIS Online         |  
  | elevation          | elevation    | 0 to 255               | ArcGIS Online         |  
  | landcover          | service class| Categorical            | MRLC                  |  
  | slope_deg          | slope degree | 0 to 90                | ArcGIS Online         |  
  | impervious         | % impervious | 0 to 100 %             | MRLC                  |  
  | soilgroup          | soil group   | Categorical            | USDA gSSURGO          |  
  | slope_as           | slope aspect | Degrees                | ArcGIS Online         |  
  | ann_prcp           | avg prcp (mm)| Float, mm              | WorldClim 2.1         |  
  | flood_den (target) | flood/cell   | Float                  | NOAA Storm Events     | 
  |--------------------|--------------|------------------------|-----------------------| 

### Data Processing

Most regions in New York have a near zero value of historic flood density so our unprocessed dataset is extremely unbalanced. Additionally, the value per pixel of historic flood density lies on a scale of 0 to about 2500 which makes the outliers (anything greater than 0 really) even more extreme. As a result, we decided to bin the flood density values into 5 discrete classes defined by equal intervals as a means of downscaling the values. After the values were binned, we transformed them back into continuous float values for regression.

Due to the inflation of zero values in our target flood density and the large size of our dataset, we were able to create a balanced dataset by downsampling so that there are near equal representations of each flood density bin. We trained and evaluated our models on both of these datasets (full and downsampled).

### Artificial Neural Network Design

We wrote the code in Python and used PyTorch as our Deep Learning framework to implement our models. We chose PyTorch because it seems to be the most popular Deep Learning framework for conducting research.
We used a Multi-Layer Perceptron (MLP) model to perform a regression on our selected explanatory variables to predict the flood density per area.  A Multi-Layer Perceptron is a feed-forward artificial neural network that uses back propagation learning for classification or regression. For this project, we will be using our MLP for regression to predict a continuous variable that has been chunked into smaller buckets. The explanatory variables will be passed into the input layer and then to the hidden layers that aim to find a multi-dimensional expansion of the input layer. The relationship between the output layer and the input layer is created by training the model on a selection of training examples and adjusting the weights of our explanatory variables (Liu, 2023).  The training was optimized using a loss function of Mean Squared Error (MSE), also known as the squared L2 norm, which finds the difference between the predicted value and real output, squares it, and averages over the batch.  To train the dataset, the data was split into 70-10-20 ratios for training, validation, and testing sets, and several architectures were trained.

### MLP for All Data

The network architecture for the first regression MLPs were fairly straightforward, but quickly caused issues in training.  These models were trained on the train set taken from all the data points, rather than the resampled.  The structure had an input layer with 7 neutrons that connected to 16, a 16 to 32 neuron hidden layer, and a 32 to 16 neuron hidden layer, and then a final output layer that led to one neuron.  Activation functions of ReL Units were between layers.  ReLU works as a piecewise function, so that values greater than zero are passed as is, and those at or below zero become zero. This means it is mostly linear and overcomes the vanishing gradient problem, making it a highly used and typical default activation function.  The loss function of the MSE was used to train the model and also assess performance in real time.  Validation sets were tested at the end of each epoch and the MSE was calculated for the train and validation sets to produce graphs.
Because of initial overfitting concerns and very long run time, we then used the same architecture with dropout regularization added.  10% of the hidden layers’ outputs were dropped to prevent possible overfitting to the training data.  Additionally, the learning rate and batch size were both adjusted to larger values in hopes of speeding up the training process from the initial 6+ hour runtimes.  Both were trained on only 10 epochs due to long run times. 

###  MLP for Resampled Data

After resampling the lower classes, especially the low-flood risk bucket of 0, to create more even class splits, the resampled data allowed for significantly faster training and validation. This is likely due to the lack of redundancy in the data, which now had less noise but enough meaningful information for the models to learn from.  We trained a similar network with the same two hidden layers on the resampled data, without any dropout regularization since we were less worried about overfitting now.  The faster run time meant a greater number of epochs was used in training, from 10 before with all data to 42 now with resampled data.
The resampling and significant training time decrease also allowed us to increase the depth of the architecture design.  We tested a network with 5 hidden layers, with multiple 28 nodes fully-connected hidden layers.



