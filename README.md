1. The orgnazation of the project

  The files needed to run main.py are all ready, and you can run main.py directly to test the model's performance. It implement PSNR and SSIM to measure the performance of Model A and B in Task A(X2) and B(X2). Besides, There are complete code for training and evaluating the models in folder "A" and "B", to run these code, the complete dataset should be downloaded to the folder "Datasets" and change the corresponding path of the dataset in the code.

2. The role of each file 

  "main.py" is used to show the performance of Model A and Model B for Task A(X2) and Task B(X2) respectively

  "model.py" contains the structure of Model A and Model B

  "DIV2K_valid_HR"contains the HR images for model testing in main.py

  "DIV2K_valid_LR_bicubic_X2/X2" contains the LR images for testing Model A in main.py

  "DIV2K_valid_LR_unknown_X2/X2" contains the LR images for testing Model B in main.py

  "A" contains the code for training and evaluating the Model A, change the path of dataset and the value of scale_factor can make the Model trained for different sub-tasks(X2,X3,X4).

  "B" contains the code for training and evaluating the Model B, change the path of dataset and the value of scale_factor can make the Model trained for different sub-tasks(X2,X3,X4).

3. the packages required to run the code

   "main.py": torch, torchvision, Pillow, glob, piq
   
   "Code for Task A(X2).py" and "Code for Task B(X2).py":numpy, torch, torchvision, Pillow, glob, piq, matplotlib
   
  
   
