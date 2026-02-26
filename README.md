## Project sturcture
Initially we did all experiments on okkkk.ipynb, but then at the end polished everything and seperated to py files, then inside inf_2 folder we continued.

Dataset was unbalanced and we initially loaded in okkk.ipynb but after we feature engineered in data manipulation.ipynb and visualized in data_visualization.ipynb

okkkkk.ipynb is experimental but there are a lot of good stuff, it is very big notebook but worth checking.





## final model
there are several models(okkk.ipynb models are not saved but good). saved models are good, but eperimental models in okkkk.ipynb are good in real-time, they have low latency.

final arhchitechture is saved in png. worth mentioning that overall we have 4-5 model architectures(+- similar but with improvements each)


- in model.py everything you need to train and related to model
- in transforms.py dataset initialization and augmentation settings, and visualization.
- fin.ipynb is orchestration of all modules and final model


## deployment
We just made simple strealit app with only file upload.
Thats all.

on streamlit our landmarks stopped working correctly

Screenshot from streamlit
![alt text](image-1.png)
![alt text](image-2.png)
![alt text](image-4.png)