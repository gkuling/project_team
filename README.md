# project_team

## Framework 

This is a package to organize, execute, and persist machine learning and applied statistical models. 

<p align="center"> <img src="./img/Framework_Diagram.PNG"  width="450" height="450">

The frame work is built of three main objects, an IO Manager, a Data Processor and a Statistical Practitioner:

- IO Manager: an object that manages all file flow in the framework, it decides where to load data from, where to save the data, and to build folders and frameworks for this loading and saving. 
- Data Processor: an object that will take an item run data type checks on it, process the data, and be able to reverse the processes to return back to the input that was given. 
- Statistical Practitioner: an object the can take data and run statistical analysis, training, or inference on that data. This object also keeps track of a model's parts, hyperparamters, and required resources to keep consistent applications. 

Between the three objects there are three key operations between two objects. 

- Transfering: includes loading data to be used by the processor and saving results that have been post processed.
- Processing: transforming input data to the proper space to be used by the practitioner, and transfering inference results from the statistical process. 
- Designing: developing and deploying a given statistical model that can have a persistent set up. The manager tells the PRactitioner where files are located and the Pracitioner checks requirements and loads data. 

## Configurations 

The currency of this framework is a configuration file. These objects take dictionaries of data that hold key aspects, or parameters to perform functions. Each object in the pro_team will have a config that it uses to understand principles necessary to do its job.

<em>An Example of a practitioner configuration that is specialized in using pytorch models. </em> 

<p align="left"> <img src="./img/PTPractitioner_config.png" height="450" >

These config files save as txt dictionary files and can easily be manually edited in notepad. These items are light weight and can give objects flexibilty to change when a donfig is loaded. 

<em>An Example of a saved UNet model configuration. We can see a very well organized dictionary saved as a text file, where individual parameters could be edited manually and the alterations would be implemented when it is reloaded.  </em> 

<p align="left"> <img src="./img/SAved_UNet_Config.png" height="450" >

This object is largely inspired by the *transformers* package from huggingface. (https://huggingface.co/transformers)

## IO Managers

Description of a Manager and their base functions 

## Data Processors

Description of Processors and their base functions 

## Statistical Practitioners 

Description of Practitioners and their base functions


