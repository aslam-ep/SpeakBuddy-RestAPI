# SpeakBuddy-RestAPI

This repository holds the flask api portion of the speak buddy application for runing the prediction on i3d model and returning the result to the application side.

1. In order to execute these code install the requirements by executing the following command.

```
$ pip install -r requirements.txt
```
2. Create a folder named Model on this path.

3. Download the model and parameter file from the google drive and place it on the Model folder.

4. Then make sure that mobile phone and your Pc is connected to the same network.

5. Then find the ip of your pc and paste this ip address on the RetrofitClient.java file of the android portion which can be found on this path:
   SpeakBudy/app/src/main/java/com/hector/speakbudy/API/
