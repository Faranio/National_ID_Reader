# National_ID_Reader
Getting a digital ID information from the national (Kazakhstan) ID card using the Telegram bot and PyTorch. Requires 4 trained models:
- IDclassification.pt
- IDregression.pt
- BackROI.pt
- FrontROI.pt

Files and their corresponding functions:
- ID.py - PyTorch dataset of ID cards
- main.py - main pipeline that performs the keypoints detection
- ROI.py - PyTorch dataset of regions of interest (ROI) in ID cards
- telegramBot.py - script that runs the Telegram Bot that receives images of ID cards
- utils.py - additional methods used in the project.
- Result.json - resultant file that shows regions on interest in Russian language.
