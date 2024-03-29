---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [✓] Create a git repository
* [✓] Make sure that all team members have write access to the github repository
* [✓] Create a dedicated environment for you project to keep track of your packages
* [✓] Create the initial file structure using cookiecutter
* [✓] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [✓] Add a model file and a training script and get that running
* [✓] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [✓] Remember to comply with good coding practices (`pep8`) while doing the project
* [✓] Do a bit of code typing and remember to document essential parts of your code
* [✓] Setup version control for your data or part of your data
* [✓] Construct one or multiple docker files for your code
* [✓] Build the docker files locally and make sure they work as intended
* [✓] Write one or multiple configurations files for your experiments
* [✓] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [✓] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [✓] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [✓] Write unit tests related to the data part of your code
* [✓] Write unit tests related to model construction and or model training
* [✓] Calculate the coverage.
* [✓] Get some continuous integration running on the github repository
* [✓] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [✓] Create a trigger workflow for automatically building your docker images
* [✓] Get your model training in GCP using either the Engine or Vertex AI
* [✓] Create a FastAPI application that can do inference using your model
* [✓] If applicable, consider deploying the model locally using torchserve
* [✓] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [✓] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [✓] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [✓] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Make sure all group members have a understanding about all parts of the project
* [ ] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

Group 60

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s220278, s232449, s233231, s222374, s233499

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

First, we utilized the PyTorch-Transformers framework (now known as pytorch_pretrained_bert). This framework, developed by HuggingFace, played a crucial role in loading the pre-trained model (bert-base-uncased) and its corresponding tokenizer. The tokenizer object allows the conversion from character strings to tokens understood by our specific model. Subsequently, we adopted the high-level framework PyTorch Lightning to streamline our model implementation and training code, benefiting from its organized and efficient structure. Then, we employed Hydra for parameter configuration management, allowing us to easily adjust and experiment with various model hyperparameters. Also, we used Wandb for logging purposes, to save metrics and variables so to check how our model was training. We also use Streamlit to create a simple web-app to use our solution. 

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We employed `pipreqs` for dependency management, automatically generating the list of dependencies and recording them in the `requirements.txt` file. To replicate our entire development environment, users should follow these steps:

1. Run `make create_environment` to create a conda environment named after the project.
2. Execute `make requirements` to install all the dependencies necessary to execute the code.
3. Optionally, users can run `make dev_requirements` to acquire the all the Developer Python Dependencies if needed.
4. To be able to run the unit tests on the code, it is also necessary to execute `make test_requirements`.

We also have a 'requirements_inference.txt'.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

From the cookiecutter template we have filled out with '.csv' files the data folder in the raw subfolder, and the tokenized data is created in the processed subfolder. The notebook folder serves as a repository for the original project notebook, which was used as a reference. Additionally, we organized the reports and test folders to contain this report and the unit tests, respectively. All the source code is located in the 'toxic_comments' folder, with the model file residing in the models subfolder and the data processor in the data subfolder. The training and prediction files are also stored in the 'toxic_comments' folder. Also there the api files are stored.
Since visualization was not incorporated into our project, the respective folder was removed. Lastly, we found it necessary to include the dockerfiles folder to store Docker files for prediction and training, and the .github/workflows folder contains files defining GitHub actions within the project repository. 

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

We have opted to follow to the standard Pep8 guidelines for code implementation, with the exception that we manually set the maximum line length to 120 characters instead of the standard 79. To ensure code consistency and adherence to these guidelines, we have integrated ruff into our workflow. This tool automatically checks and applies any necessary formatting corrections to the code with each pull request made to the main branch. 

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

In total, we have implemented 3 tests for the data loading part and 5 for the model. Regarding the data, we ensure that we load all the datapoints, check if the shape corresponds to the expected one, and verify the presence of all labels for the training data in the dataset. For the model, our tests include validating the correct shape of the output, ensuring the predicted values are binary, verifying the accurate computation of loss and gradients, and confirming that the model can be saved and loaded without encountering errors.

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

![my_image](figures/coverage.png)

We obtained 100% on the coverage of the data part, where we actually analyze the structure of the results of the *make_dataset.py* without analyzing the file itself. We obtain 93% in the code that tests the model which may suggest that few lines are redundant as they are not executed and this means the code itself could be optimized. The model only reaches 40% coverage which suggest that many parts of it are not tested. This however doesn't guarantee anything as even a 100% coverage wouldn't guarantee the code to be correct however it would increase the probability of that. We also wanted to note that one of the tests couldn't be executed as the memory couldn't handle it.
### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We used both branches and pull requests. We created a list of issues with all the tasks to be done and when we assigned one to a team member they would create a local branch connected to the issue and work on that. When the work was done and pushed the person would create a pull request, the codecheck would run with ruff, and someone else on the team had to approve the solution, and merge into main. We however allowed team members to make changes directly on main to allow for faster small modifications although we know this is not the safer option.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

Yes, we implemented data version control, we tested both google drive and google cloud storage as remotes. In particular, it helped us to easily disseminate our training, testing data and models to different machines.

For the scope of our model, the version control part of DVC was probably not as important, as our project was not large enough to go through several iterations of our data.

However, we faced severe difficulties in integrating dvc into our github actions for model and data testing. The reason is, that neither authentication with google drive is seemingly not possible in an automated manner, and dvc-gs expects a credential file, but we didn't find a way to supply in Github actions.

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

For CI we are automatically checking code smells using ruff, as well as automatically formatting the code using Github actions. Due to authentication problems mentioned above, we were unable to run the majority of our tests in Github actions.
We also test the Python version and the requirements and we cache pip dependencies. All the tests are computed both for Ubuntu and macOS systems.

For continuous delivery, we set up three containers that get automatically built and some are also automatically deployed. The three containers we have are one container for training, one with a streamlit app over prediction, and one hosting an API with Flask.

Model data necessary for running inference is pulled from a cloud storage bucket. No automatic update of the models included in the containers is set up to avoid accidentally using a bad model in production due to a new training run that used bad parameters.

The streamlit container automatically gets deployed to google cloud run after is has been built and can be reached using the URL: <https://inference-streamlit-kjftsv3ocq-ez.a.run.app>.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

For reproduceability and tracking of experiments we are using model configurations and wandb for logging.

Whenever a model is being trained, the training logs containing the model configuration, training configuration, loss, performance, etc. get sent to wandb. This allows us to keep track of how a model performed with a given configuration. 

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We use hydra to load our configuration files for the model and training. Hydra allows us to pass command-line arguments to override the configuration. Any overrides also get logged using wandb. For a default run configuration, no overrides have to be made.

Example execution: python3 ./toxic_comments/train_model.py param_to_override_optional="some value"

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We have implemented a robust system from the beginning for ensuring the reproducibility and version control of our experiments. We used multiple of the config files incorporated with Hydra to make our experiments reproducible. Indeed, together with the model also the hyperparameters are saved thus allowing for reproducibility and version control. Also W&B helped us in keep our experiments reproducibles as it saves the hyperparameters together with the logs. Therefore we are sure that no information is lost even if running multiple experiments. Indeed, this approach ensures that every facet of our experiments is systematically documented, allowing for reproducibility across different runs.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

![Wandb](figures/screenshot_wandb.png)
![Wandb run overview](figures/screenshot_wandb_run_overview.png)

As seen in the images, we have used Weights and biases to keep track of experiments. We decided on wand because it easily allowed all team members to have access to logs & metrics of past experiments. Additionally, wandb allowed us to do sweeps and keep track of those as well. The first image shows the overview over all past experiments that were run, while the second image show the system configuration used for running an experiment in the cloud.

We tracked the following metrics:

* Validation Loss: Indicates the model's prediction error on unseen data, crucial for detecting overfitting.
* Train Loss: Shows the model's error on training data, essential for understanding how well the model is learning.
* Test Accuracy: Reflects the percentage of correct predictions on new data, vital for assessing model generalization.
* Test Precision: Measures the proportion of true positives among positive predictions, important where false positives have high costs.
* Loss: Aggregated training loss, useful for tracking overall learning progress.
* Configuration: The exact exact configuration that was used for training is key for reproduceability of experiments
* System parameters: useful for finding bottleneck or errors

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

For our project we made 3 different docker containers. One for training the model and the other two to run inference. 
For training the model initially we ran into issues with docker containers working with VertexAI and we could not use it to train the model. For training we cloned the repository locally over a VM running on the gcp instance and conducted training. Our effort was to utilize docker and VertexAI to train and run the model, but as explained earlier training did not happen using that approach and our docker kept running into issues with cloud atrifact registry. .However we were able to host both the inference containers using docker containers made using cloud build. 

Streamlit - ENTRYPOINT ["streamlit", "run", "./toxic_comments/api/streamlit_input_inference.py"]
Flask - ENTRYPOINT ["python", "./toxic_comments/slowapi/ask.py"]
 
### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

To improve our code, we used a mix of tools and techniques. We used logging to get information on the code. We used VSCode debugging with breakpoints, and strategically placed print statements to extract valuable insights as well. Our debugging efforts were further augmented by the integration of continuous integration through Ruff to check for untracked mistakes and correct them. We did not use profiling as we focus on having as many parts working as possible of the ideal pipeline even if that was not optimized. Although we did not extensively use profiling, our goal was to ensure the pipeline's functionality before fine-tuning for efficiency. 

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We used the following gcp services:

* Cloud storage: storage of training/testin data and trained models
* Container registry: Storage of built containers
* Compute engine: Virtual machines For training of models
* Cloud build: Building and deploying docker containers
* Cloud run: Hosting of Application using our model. Automatic deployment of updates
 vial cloud build

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We used the compute engine to train our models. The machine we used was an n-1-standard-8 machine with one nvidia v100 in the zone eu-west-4a. We used the deep learning on linux image "Deep Learning VM with CUDA 12.1 M115" so we had some dependencies and the nvidia drivers pre-installed. To save on VM-costs we decided to use spot instances, as our machine would not have to run for long times and it would not be a big issue if it got stopped during one of our training runs.

For training, we did not end up using our training container because we ran into authentication issues and it was
simply faster to clone our repository than figuring out authentication. However, we tried our docker containers
locally.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

![my_image](figures/bucket_cloud.png)

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

![my_image](figures/registry_cloud.png)

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

![my_image](figures/build_cloud.png)

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We delpoyed our model using 2 approaches, one using a streamlit app and other other using a flask API. Both the methods deploy the model over gcp and have similar interface. The streamlit app offers a text input and then passes it to the predict fucntion to run inference, the output is the probabilites for each of the 6 classes - toxic, severe_toxic, obscene, threat, insult, identity_hate model is trained to classify for. The flask API does something similar with the text input, but also offers running inference on text files. The output is again probabilites of the 6 classes model is trained on. <br>

Deploying models from both approaches followed standard procedure to make a .yaml file first and the creating a cloud build trigger over gcp. Both the trigger actions are associated with their respective .yaml files and the rest is taken care by the cloud. The template provided during the course is used works well after we edit the project ID.

![flask](/reports/figures/flask.png) <br>
![streamlit](/reports/figures/streamlit.png)


### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We played around with data drifting as it can have an impact on the performances of the model as the frequency, and sometimes meaning, of some words can change over time thus yelding unexpected results by our toxic comment classifier. However, we did not implemented monitoring, although we took a look at the default metrics on Google cloud run of our streamlit application. We thought it would have been nice though to have some alerts that can monitor the amount of requests and utilization to make sure there won't be problems. Probably, however, the most important ones would have been for application errors. Ultimately we decided not to merge to code for data driting to the main branch but the html reports and tests can be seen in the data_drift branch.

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>  
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

$50 credits were alloted to use on gcp platform and towards the end of the project we are now left with $2.44. Overall we ended up using $47.56 while working on our project utilizing VM services and data storage. Most of our credits were utilized training the model and accesing the buckets for data storage. We ended up losing 23 and 18 credits over one weekend as we forgot to turn off the VM and that service accounted for the most expensive way we spent our credits. Given the scope of the project our estimate goal would have been to use around 8-10 credits including training and accesing data buckets. 


## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

![scheme](/reports/figures/scheme.png)
The diagram starts with our local setup, where we developed our application using PyTorch and PyTorch Transformer, adhering to the PyTorch Lightning paradigm. Locally, we managed the project-specific dependencies through Conda environments. The project follow the structure of the coockiecutter template.
Whenever we push new code to GitHub and create a pull request to the main branch, GitHub Actions are triggered and validate the code using codecheck workflow, both for mac-os that ubuntu system. If formatting errors occur, ruff corrects and autocommits the changes required. After passing the tests, peer review is required before being able to merge in the main.
Subsequently, completing the push of new code to the main branch triggers a cloud build that build and deploy the docker containers, which are then stored in the container registry.
DVC manages data and model versions, which are stored in a GCP bucket. 
The compute engine interacts with the GCP bucket, loading training data and saving the obtained model.
End users interact with the application via API requests, wherein Cloud Run deploys the optimal model, executes interferences and return the results.



### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

The first problem we had was that the code we used as a reference was using FastAI v1 which is now deprecated and moving it to v2 was not trivial. We actually found a second solution and we started from that. We had some struggles to make different services and different computers systems work together, although ultimately we found out how to deal with them. Another issue we had was when some modifications, althogh made in branches and worked, once merged in the main we had other parts that were not working anymore; we thus spent some time to fix this issues. We also spent some time working on the cloud because the feedback is difficult and when changing something on cloudbuild can't be checked locally so they require the change to be pushed before being able to evaluate it, as well as issues with IAM permissions. The main problem of the project was that the predictor was only working correctly with one word, the problem was in how the predict file was handling data inputs, after debugging and closely analyzing the code we found the issue and solved it. 

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

We all worked in finding the problem and organizing the work, we also all contributed to help each other and fix mistakes, therefore we'll just write below who first created a part of the project and so was more involved in that.

s220278: Worked with the cloud in data storage, training, deploying it. Also added CI to the cloud and wandb. Created streamlit.
s232449: Implemented the predict script and make dataset, added continuous integration and documentation.
s233231: Implemented tests and coverage, pruning and quantization, and organized outputs and dvc.
s222374: Implemented dockers and triggers. Worked on the deployment of Flask api. Implemented dvc. 
s233499: Implemented model and training script, worked on training on the cloud. Created checks for data drifting.
