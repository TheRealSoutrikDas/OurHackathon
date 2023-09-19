# OurHackathon
Python Must be installed
1. We need to make the virtual environment, type >> python -m venv .venv (For Windows) and $ python3 -m venv .venv (For Mac or linux)
2. Install Numpy: >> pip install numpy
3. Install nltk: >> pip install nltk
4. Install PyTorch: Windows User with GPU: >> pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
                    Windows User Without GPU: >> pip3 install torch torchvision torchaudio
                    Mac Users: $ pip3 install torch torchvision torchaudio
5. Install Flask: >> pip install flask
6. Install Flask_CORS: >> pip install flask-cors
7. type >> .venv\Scripts\activate
8. to train the model run "python train.py"
9. To run the chat run "python chat.py"
10. To run web app run "python app.py"
   [For ERROR in step 8 or 9 or 10: If it shows error open nltk_util.py remove the # from line 3 run "nltk_util.py". after it is done downloading punkt comment out line 3 using "#" then repeat step 8 or 9 or 10]
11. open localhost:5000 in browser to open website.
    
